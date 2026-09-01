from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from heritage_restore.config import ProcessingConfig
from heritage_restore.export import json_safe_measurements, result_zip_bytes
from heritage_restore.io import ImageValidationError, image_to_png_bytes, load_image_bytes, mask_to_png_bytes
from heritage_restore.pipeline import ProcessingResult, RestorationPipeline


st.set_page_config(page_title="HeritageGuard", page_icon="🏛️", layout="wide")


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image


def reset_application() -> None:
    for key in list(st.session_state):
        del st.session_state[key]
    st.rerun()


def configuration_panel() -> ProcessingConfig:
    st.sidebar.header("Processing controls")
    st.sidebar.caption("All values are applied to both single-image and batch workflows.")
    clahe = st.sidebar.slider("Contrast (CLAHE clip)", 0.5, 8.0, 1.5, 0.1)
    illumination = st.sidebar.slider("Illumination kernel", 15, 251, 151, 2)
    denoise = st.sidebar.selectbox("Noise filter", ["auto", "median", "gaussian"])
    crack_kernel = st.sidebar.slider("Crack detector scale", 5, 41, 17, 2)
    block_size = st.sidebar.slider("Adaptive threshold block", 11, 71, 31, 2)
    adaptive_c = st.sidebar.slider("Adaptive threshold C", 1, 21, 7, 1)
    sensitivity = st.sidebar.slider("Stain sensitivity", 0.5, 6.0, 2.7, 0.1)
    minimum_area = st.sidebar.slider("Minimum damage area (px)", 0, 500, 18, 1)
    inpaint_radius = st.sidebar.slider("Restoration radius", 1, 15, 3, 1)
    rotation = st.sidebar.slider("Rotation correction", -15.0, 15.0, 0.0, 0.5)

    st.sidebar.subheader("Region of interest")
    use_roi = st.sidebar.checkbox("Process a selected ROI")
    if use_roi:
        roi_x = st.sidebar.slider("ROI left (%)", 0, 90, 0) / 100.0
        roi_y = st.sidebar.slider("ROI top (%)", 0, 90, 0) / 100.0
        max_width = max(10, int((1.0 - roi_x) * 100))
        max_height = max(10, int((1.0 - roi_y) * 100))
        roi_width = st.sidebar.slider("ROI width (%)", 10, max_width, max_width) / 100.0
        roi_height = st.sidebar.slider("ROI height (%)", 10, max_height, max_height) / 100.0
    else:
        roi_x = roi_y = 0.0
        roi_width = roi_height = 1.0
    if st.sidebar.button("Reset application", width="stretch"):
        reset_application()
    return ProcessingConfig(
        clahe_clip_limit=clahe,
        illumination_kernel=illumination,
        denoise_method=denoise,
        crack_kernel=crack_kernel,
        adaptive_block_size=block_size,
        adaptive_c=adaptive_c,
        stain_sensitivity=sensitivity,
        minimum_region_area=minimum_area,
        inpaint_radius=inpaint_radius,
        rotation_degrees=rotation,
        roi_x=roi_x,
        roi_y=roi_y,
        roi_width=roi_width,
        roi_height=roi_height,
    )


def display_metadata(metadata: dict[str, object]) -> None:
    with st.expander("Validated image information", expanded=False):
        columns = st.columns(4)
        fields = [
            ("Dimensions", f"{metadata['width']} x {metadata['height']} px"),
            ("Channels", str(metadata["channels"])),
            ("Bit depth", f"{metadata['bit_depth']}-bit"),
            ("File size", f"{int(metadata['file_size_bytes']) / 1024:.1f} KB"),
            ("Format", str(metadata["format"])),
            ("Selected region", str(metadata["selected_region"])),
        ]
        for index, (label, value) in enumerate(fields):
            columns[index % len(columns)].metric(label, value)


def display_measurements(result: ProcessingResult) -> None:
    values = result.measurements
    st.subheader("Domain decision and measurable output")
    decision = str(values["decision"])
    if decision.startswith("PASS"):
        st.success(decision)
    else:
        st.warning(decision)
    columns = st.columns(5)
    columns[0].metric("Damage area", f"{values['damage_area_percent']:.2f}%")
    columns[1].metric("Detected regions", int(values["detected_region_count"]))
    columns[2].metric("Restoration score", f"{values['restoration_quality_score']:.1f}/100")
    columns[3].metric("Contrast change", f"{values['contrast_improvement']:+.2f}")
    columns[4].metric("Processing time", f"{values['processing_time_ms']:.0f} ms")
    if "segmentation_iou" in values:
        evaluation = st.columns(5)
        evaluation[0].metric("IoU", f"{values['segmentation_iou']:.3f}")
        evaluation[1].metric("Precision", f"{values['segmentation_precision']:.3f}")
        evaluation[2].metric("Recall", f"{values['segmentation_recall']:.3f}")
        evaluation[3].metric("F1", f"{values['segmentation_f1']:.3f}")
        evaluation[4].metric("Accuracy", f"{values['segmentation_accuracy']:.3f}")
    if "psnr_after_db" in values:
        quality = st.columns(4)
        quality[0].metric("PSNR before", f"{values['psnr_before_db']:.2f} dB")
        quality[1].metric("PSNR after", f"{values['psnr_after_db']:.2f} dB", f"{values['psnr_improvement_db']:+.2f}")
        quality[2].metric("SSIM before", f"{values['ssim_before']:.3f}")
        quality[3].metric("SSIM after", f"{values['ssim_after']:.3f}", f"{values['ssim_improvement']:+.3f}")

    with st.expander("Complete structured measurements"):
        st.dataframe(
            [{"measurement": key, "value": str(value)} for key, value in json_safe_measurements(values).items()],
            width="stretch",
            hide_index=True,
        )


def display_pipeline(result: ProcessingResult) -> None:
    st.subheader("Integrated processing workflow")
    first = st.columns(3)
    first[0].image(bgr_to_rgb(result.roi_image), caption="1. Validated input / ROI", width="stretch")
    first[1].image(bgr_to_rgb(result.enhanced), caption="2. Illumination + CLAHE enhancement", width="stretch")
    first[2].image(bgr_to_rgb(result.overlay), caption="3. Damage analysis overlay", width="stretch")
    second = st.columns(3)
    second[0].image(result.crack_mask, caption="Crack mask: threshold + edges + morphology", width="stretch")
    second[1].image(result.stain_mask, caption="Stain mask: Lab color segmentation", width="stretch")
    second[2].image(bgr_to_rgb(result.restored), caption="4. Restored result (Telea inpainting)", width="stretch")
    if np.any(result.reference_difference_mask):
        st.image(
            result.reference_difference_mask,
            caption="Reference-assisted difference mask (cyan regions in overlay; used for restoration when a clean reference is supplied)",
            width="stretch",
        )

    with st.expander("Color spaces, filter comparison, and feature evidence"):
        color_columns = st.columns(4)
        color_columns[0].image(result.color_spaces["grayscale"], caption="Grayscale")
        color_columns[1].image(cv2.cvtColor(result.color_spaces["lab"], cv2.COLOR_LAB2RGB), caption="Lab")
        color_columns[2].image(cv2.cvtColor(result.color_spaces["hsv"], cv2.COLOR_HSV2RGB), caption="HSV")
        color_columns[3].image(result.color_spaces["binary"], caption="Otsu binary")
        filter_columns = st.columns(2)
        filter_columns[0].image(result.filter_comparison["gaussian"], caption="Gaussian filter comparison")
        filter_columns[1].image(result.filter_comparison["median"], caption="Median filter comparison")
        st.caption(
            f"Selected filter: **{result.measurements['selected_filter']}**. Automatic selection compares noise removal and edge retention."
        )


def single_workflow(config: ProcessingConfig) -> None:
    st.markdown("### Single image analysis")
    upload_columns = st.columns(3)
    uploaded = upload_columns[0].file_uploader(
        "Damaged image (required)", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], key="single_input"
    )
    reference_upload = upload_columns[1].file_uploader(
        "Clean reference (optional, enables PSNR/SSIM)", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], key="reference"
    )
    mask_upload = upload_columns[2].file_uploader(
        "Ground-truth damage mask (optional)", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], key="truth_mask"
    )
    if uploaded is None:
        st.info("Upload a heritage photograph, document, painting, artifact image, or inscription to begin.")
        return
    try:
        image, metadata = load_image_bytes(uploaded.getvalue(), uploaded.name)
        display_metadata(metadata.as_dict())
        if st.button("Process image", type="primary", width="stretch"):
            reference = load_image_bytes(reference_upload.getvalue(), reference_upload.name)[0] if reference_upload else None
            truth_mask = load_image_bytes(mask_upload.getvalue(), mask_upload.name)[0] if mask_upload else None
            with st.spinner("Running enhancement, segmentation, feature extraction, restoration, and evaluation..."):
                st.session_state["single_result"] = RestorationPipeline(config).process(image, reference, truth_mask)
                st.session_state["single_name"] = Path(uploaded.name).stem
                st.session_state["single_metadata"] = metadata.as_dict()
        result = st.session_state.get("single_result")
        if result is None:
            st.image(bgr_to_rgb(image), caption="Validated original", width="stretch")
            return
        display_pipeline(result)
        display_measurements(result)
        stem = st.session_state.get("single_name", "heritage_image")
        download_columns = st.columns(4)
        download_columns[0].download_button(
            "Download restored PNG", image_to_png_bytes(result.restored), f"{stem}_restored.png", "image/png", width="stretch"
        )
        download_columns[1].download_button(
            "Download damage mask", mask_to_png_bytes(result.damage_mask), f"{stem}_damage_mask.png", "image/png", width="stretch"
        )
        download_columns[2].download_button(
            "Download metrics JSON",
            json.dumps(json_safe_measurements(result.measurements), indent=2),
            f"{stem}_metrics.json",
            "application/json",
            width="stretch",
        )
        download_columns[3].download_button(
            "Download complete ZIP",
            result_zip_bytes(result, stem, st.session_state.get("single_metadata")),
            f"{stem}_heritageguard_results.zip",
            "application/zip",
            width="stretch",
        )
    except (ImageValidationError, ValueError, RuntimeError) as exc:
        st.error(f"Input could not be processed: {exc}")


def batch_zip(files: list, config: ProcessingConfig) -> tuple[bytes, list[dict[str, object]]]:
    archive_buffer = io.BytesIO()
    summary_rows: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for uploaded in files:
            try:
                image, metadata = load_image_bytes(uploaded.getvalue(), uploaded.name)
                result = RestorationPipeline(config).process(image)
                stem = Path(uploaded.name).stem
                archive.writestr(f"{stem}/{stem}_restored.png", image_to_png_bytes(result.restored))
                archive.writestr(f"{stem}/{stem}_damage_overlay.png", image_to_png_bytes(result.overlay))
                archive.writestr(f"{stem}/{stem}_damage_mask.png", mask_to_png_bytes(result.damage_mask))
                archive.writestr(
                    f"{stem}/{stem}_metrics.json", json.dumps(json_safe_measurements(result.measurements), indent=2)
                )
                row: dict[str, object] = {"filename": uploaded.name, "status": "processed"}
                row.update(metadata.as_dict())
                row.update(json_safe_measurements(result.measurements))
                summary_rows.append(row)
            except (ImageValidationError, ValueError, RuntimeError) as exc:
                summary_rows.append({"filename": uploaded.name, "status": "failed", "error": str(exc)})
        fieldnames: list[str] = []
        for row in summary_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
        archive.writestr("batch_summary.csv", csv_buffer.getvalue())
    return archive_buffer.getvalue(), summary_rows


def batch_workflow(config: ProcessingConfig) -> None:
    st.markdown("### Batch image analysis")
    st.caption("Select every image in a folder. The CLI also accepts a folder path directly.")
    uploads = st.file_uploader(
        "Image folder / multiple images",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="batch_inputs",
    )
    if not uploads:
        st.info("Choose two or more images for batch processing and CSV export.")
        return
    if st.button("Process batch", type="primary", width="stretch"):
        progress = st.progress(0, text="Starting batch...")
        with st.spinner("Processing image folder..."):
            data, rows = batch_zip(uploads, config)
            st.session_state["batch_zip"] = data
            st.session_state["batch_rows"] = rows
        progress.progress(100, text="Batch complete")
    rows = st.session_state.get("batch_rows")
    if rows:
        processed = sum(row.get("status") == "processed" for row in rows)
        failed = len(rows) - processed
        columns = st.columns(2)
        columns[0].metric("Processed", processed)
        columns[1].metric("Failed", failed)
        st.dataframe(rows, width="stretch", hide_index=True)
        st.download_button(
            "Download batch outputs + CSV",
            st.session_state["batch_zip"],
            "heritageguard_batch_results.zip",
            "application/zip",
            type="primary",
            width="stretch",
        )


def evidence_page() -> None:
    st.markdown("### Method and rubric evidence")
    st.markdown(
        """
        **Specific problem:** detect cracks and stains in digitized old photographs, restore detected regions, and quantify the result.

        **Integrated workflow:** validation and metadata → ROI/rotation → Grayscale/Lab/HSV conversion → illumination correction and CLAHE → Gaussian/median comparison → Otsu/adaptive thresholding → Canny edges and contours → morphological and Lab region segmentation → shape/area/texture features → Telea restoration → decision, score, export, and evaluation.

        The downloadable output contains the images, parameters, and structured measurements needed to reproduce each result. See `docs/RUBRIC_TRACEABILITY.md` and `docs/VIVA_GUIDE.md` for the complete evidence map and explanation prompts.
        """
    )
    st.warning(
        "Academic-integrity checkpoint: the bundled 30-image dataset is synthetic demonstration data. Add and certify at least ten images you personally captured or produced before submission."
    )


st.title("HeritageGuard")
st.caption("Old-photograph damage detection, measurable restoration, and evaluation — Python + OpenCV")
config = configuration_panel()
single_tab, batch_tab, evidence_tab = st.tabs(["Single image", "Batch / folder", "Rubric evidence"])
with single_tab:
    single_workflow(config)
with batch_tab:
    batch_workflow(config)
with evidence_tab:
    evidence_page()
