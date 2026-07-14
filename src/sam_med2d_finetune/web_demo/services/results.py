from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import cv2
import nibabel as nib
import numpy as np

from sam_med2d_finetune.web_demo.config import (
    DEFAULT_DEMO_MODE,
    GENERATED_IMAGE_DIR,
    OUTPUTS_DIR,
    PIPELINE_SUMMARY_FILE,
    POST_MASK_FILES,
    RAW_MASK_FILES,
    SUMMARY_JSON_CANDIDATES,
    SUMMARY_MD_CANDIDATES,
    VIEWER_CANDIDATES,
    ensure_web_demo_dirs,
    get_demo_mode,
    normalize_demo_mode,
    resolve_sample_result_mode,
)
from sam_med2d_finetune.inference.visualize import render_case


COMBINED_MASK_COLORS = {
    1: np.array([42, 157, 143], dtype=np.uint8),
    2: np.array([69, 123, 157], dtype=np.uint8),
    4: np.array([230, 57, 70], dtype=np.uint8),
}

CLASS_MASK_CANDIDATES = {
    "WT": ("post_WT.nii.gz", "WT.nii.gz"),
    "TC": ("post_TC.nii.gz", "TC.nii.gz"),
    "ET": ("post_ET.nii.gz", "ET.nii.gz"),
}

DEFAULT_VIEWER_MASK_BY_MODE = {
    "standard": "WT",
    "multiclass": "all",
}

MULTICLASS_VIEWER_MASKS = ("WT", "TC", "ET")


def encode_result_id(result_dir: Path) -> str:
    relative_path = result_dir.resolve().relative_to(OUTPUTS_DIR.resolve())
    encoded = base64.urlsafe_b64encode(str(relative_path).encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")


def decode_result_id(result_id: str) -> Path:
    padding = "=" * (-len(result_id) % 4)
    relative_text = base64.urlsafe_b64decode((result_id + padding).encode("ascii")).decode("utf-8")
    candidate = (OUTPUTS_DIR / relative_text).resolve()
    candidate.relative_to(OUTPUTS_DIR.resolve())
    if not candidate.is_dir():
        raise FileNotFoundError(f"结果目录不存在: {candidate}")
    return candidate


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_text(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _detect_result_mode(
    result_dir: Path,
    *,
    case_meta: dict[str, Any] | None,
    pipeline_summary: dict[str, Any] | None,
) -> str | None:
    meta_mode = ((case_meta or {}).get("web_demo_mode") or {}).get("key")
    if meta_mode:
        return normalize_demo_mode(meta_mode)

    pipeline_mode = ((pipeline_summary or {}).get("mode") or {}).get("key")
    if pipeline_mode:
        return normalize_demo_mode(pipeline_mode)

    sample_mode = resolve_sample_result_mode(result_dir)
    if sample_mode:
        return normalize_demo_mode(sample_mode)
    return None


def resolve_result_mode(
    result_dir: Path,
    *,
    requested_mode: str | None,
    case_meta: dict[str, Any] | None,
    pipeline_summary: dict[str, Any] | None,
) -> str:
    requested = normalize_demo_mode(requested_mode) if requested_mode else None
    actual_mode = _detect_result_mode(
        result_dir,
        case_meta=case_meta,
        pipeline_summary=pipeline_summary,
    )
    if requested and actual_mode and requested != actual_mode:
        if requested == "multiclass":
            raise FileNotFoundError("\u8be5\u75c5\u4f8b\u6682\u672a\u751f\u6210\u591a\u7c7b\u522b\u5206\u6790\u7ed3\u679c\u3002")
        raise FileNotFoundError("\u5f53\u524d\u7ed3\u679c\u4e0d\u5c5e\u4e8e\u6807\u51c6\u6a21\u5f0f\u3002")
    if actual_mode:
        return actual_mode
    if requested:
        return requested
    return DEFAULT_DEMO_MODE


def _iter_search_dirs(result_dir: Path):
    current = result_dir.resolve()
    outputs_root = OUTPUTS_DIR.resolve()
    while True:
        yield current
        if current == outputs_root:
            break
        if outputs_root not in current.parents:
            break
        current = current.parent


def _find_nearest_file(result_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    for directory in _iter_search_dirs(result_dir):
        for filename in candidates:
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    return None


def _normalize_viewer_mask(mask_name: str | None, mode_key: str) -> str:
    normalized_mode = normalize_demo_mode(mode_key)
    if normalized_mode == "standard":
        return "WT"

    raw_value = str(mask_name or DEFAULT_VIEWER_MASK_BY_MODE[normalized_mode]).strip()
    if not raw_value or raw_value.lower() == "all":
        return "all"

    requested_masks = {
        item.strip().upper()
        for item in raw_value.replace("+", ",").split(",")
        if item.strip()
    }
    valid_masks = [mask for mask in MULTICLASS_VIEWER_MASKS if mask in requested_masks]
    if len(valid_masks) == len(MULTICLASS_VIEWER_MASKS):
        return "all"
    if not valid_masks:
        return DEFAULT_VIEWER_MASK_BY_MODE[normalized_mode]
    if len(valid_masks) == 1:
        return valid_masks[0]
    return ",".join(valid_masks)


def _viewer_mask_to_slug(mask_name: str) -> str:
    normalized = str(mask_name).strip()
    if normalized.lower() == "all":
        return "all"
    if normalized.lower() == "combined":
        return "combined"
    parts = [item.strip().lower() for item in normalized.split(",") if item.strip()]
    return "_".join(parts) if parts else "all"


def _viewer_filename_candidates(mask_name: str) -> tuple[str, ...]:
    slug = _viewer_mask_to_slug(mask_name)
    return (f"preview_3d_compare_{slug}.html", f"preview_3d_{slug}.html")


def _can_render_viewer(result_dir: Path, mask_name: str) -> bool:
    requested_masks = MULTICLASS_VIEWER_MASKS if mask_name == "all" else tuple(mask_name.split(","))
    return all(
        any((result_dir / filename).is_file() for filename in CLASS_MASK_CANDIDATES[mask_name_item])
        for mask_name_item in requested_masks
    )


def _ensure_viewer_file(result_dir: Path, mask_name: str) -> Path | None:
    for filename in _viewer_filename_candidates(mask_name):
        candidate = result_dir / filename
        if candidate.is_file():
            return candidate

    if not _can_render_viewer(result_dir, mask_name):
        return None

    save_path = result_dir / f"preview_3d_compare_{_viewer_mask_to_slug(mask_name)}.html"
    try:
        render_case(output_dir=result_dir, mask_name=mask_name, save_path=save_path)
    except Exception:
        return None
    return save_path if save_path.is_file() else None


def find_viewer_file(
    result_dir: Path,
    mode_key: str | None = None,
    *,
    mask_name: str | None = None,
    auto_generate: bool = False,
) -> Path | None:
    if mode_key is None and mask_name is None:
        generic_candidates = (
            "preview_3d_compare_all.html",
            "preview_3d_all.html",
            "preview_3d_compare_WT.html",
            "preview_3d_WT.html",
            "preview_3d_compare_TC.html",
            "preview_3d_TC.html",
            "preview_3d_compare_ET.html",
            "preview_3d_ET.html",
            *VIEWER_CANDIDATES,
        )
        for filename in generic_candidates:
            candidate = result_dir / filename
            if candidate.is_file():
                return candidate
        return None

    resolved_mode = normalize_demo_mode(mode_key)
    resolved_mask = _normalize_viewer_mask(mask_name, resolved_mode)
    for filename in _viewer_filename_candidates(resolved_mask):
        candidate = result_dir / filename
        if candidate.is_file():
            return candidate
    if not auto_generate:
        return None
    return _ensure_viewer_file(result_dir, resolved_mask)


def _find_mask_path(result_dir: Path) -> Path | None:
    for filename in ("post_combined_label.nii.gz", "combined_label.nii.gz"):
        candidate = result_dir / filename
        if candidate.is_file():
            return candidate
    return None


def _load_volume(path: Path | None) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    try:
        return np.asarray(nib.load(str(path)).dataobj)
    except Exception:
        return None


def _load_nifti_data_and_spacing(path: Path | None) -> tuple[np.ndarray | None, tuple[float, float, float] | None]:
    if path is None or not path.is_file():
        return None, None
    try:
        image = nib.load(str(path))
        data = np.asarray(image.dataobj)
        zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    except Exception:
        return None, None

    if len(zooms) < 3 or any(value <= 0 for value in zooms[:3]):
        return data, None
    return data, (zooms[0], zooms[1], zooms[2])


def _normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32, copy=False)
    mask = volume != 0
    if not np.any(mask):
        return np.zeros_like(volume, dtype=np.uint8)

    min_value = float(volume[mask].min())
    max_value = float(volume[mask].max())
    scale = max(max_value - min_value, 1e-8)
    normalized = np.zeros_like(volume, dtype=np.float32)
    normalized[mask] = (volume[mask] - min_value) / scale
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def _pick_slice_indices(mask_volume: np.ndarray | None, depth: int) -> list[int]:
    if depth <= 0:
        return []
    if mask_volume is None or mask_volume.ndim != 3:
        center = depth // 2
        return sorted({max(0, center - 8), center, min(depth - 1, center + 8)})

    area_by_slice = mask_volume.astype(bool).sum(axis=(0, 1))
    if int(np.max(area_by_slice)) <= 0:
        center = depth // 2
        return sorted({max(0, center - 8), center, min(depth - 1, center + 8)})

    center = int(np.argmax(area_by_slice))
    return sorted({max(0, center - 8), center, min(depth - 1, center + 8)})


def _overlay_mask(base_slice: np.ndarray, mask_slice: np.ndarray | None) -> np.ndarray:
    rgb = np.stack([base_slice, base_slice, base_slice], axis=-1)
    if mask_slice is None:
        return rgb

    overlay = rgb.copy()
    for label_value, color in COMBINED_MASK_COLORS.items():
        current_mask = mask_slice == label_value
        if not np.any(current_mask):
            continue
        overlay[current_mask] = (0.55 * overlay[current_mask] + 0.45 * color).astype(np.uint8)
    return overlay


def _build_static_url(path: Path) -> str:
    relative_path = path.resolve().relative_to(GENERATED_IMAGE_DIR.parent.resolve())
    return "/static/" + str(relative_path).replace("\\", "/")


def _generate_slice_gallery(result_dir: Path, result_id: str, case_meta: dict[str, Any] | None) -> list[dict[str, str]]:
    ensure_web_demo_dirs()
    target_dir = GENERATED_IMAGE_DIR / result_id
    target_dir.mkdir(parents=True, exist_ok=True)

    cached_images = sorted(target_dir.glob("*.png"))
    if cached_images:
        return [{"label": image_path.stem.replace("_", " "), "url": _build_static_url(image_path)} for image_path in cached_images]

    modality_paths = (case_meta or {}).get("modality_paths", {})
    preferred_image_path = None
    for modality in ("flair", "t1ce", "t2", "t1"):
        raw_path = modality_paths.get(modality)
        if raw_path:
            preferred_image_path = Path(raw_path)
            break

    image_volume = _load_volume(preferred_image_path)
    if image_volume is None or image_volume.ndim != 3:
        return []

    mask_volume = _load_volume(_find_mask_path(result_dir))
    image_volume = _normalize_volume(image_volume)
    slice_indices = _pick_slice_indices(mask_volume, image_volume.shape[2])

    images: list[dict[str, str]] = []
    for slice_index in slice_indices:
        base_slice = image_volume[:, :, slice_index]
        mask_slice = mask_volume[:, :, slice_index] if mask_volume is not None else None

        raw_path = target_dir / f"z{slice_index:03d}_raw.png"
        overlay_path = target_dir / f"z{slice_index:03d}_overlay.png"

        raw_rgb = np.stack([base_slice] * 3, axis=-1)
        overlay_rgb = _overlay_mask(base_slice, mask_slice)
        cv2.imwrite(str(raw_path), cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

        images.extend(
            [
                {"label": f"z={slice_index} 原始切片", "url": _build_static_url(raw_path)},
                {"label": f"z={slice_index} 叠加结果", "url": _build_static_url(overlay_path)},
            ]
        )
    return images


def _format_float(value: Any) -> str | None:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return None


def _format_int(value: Any) -> str | None:
    try:
        return f"{int(value):,}"
    except Exception:
        return None


def _normalize_spacing(spacing: Any) -> tuple[float, float, float] | None:
    if not isinstance(spacing, (list, tuple)) or len(spacing) < 3:
        return None
    try:
        normalized = tuple(float(spacing[index]) for index in range(3))
    except Exception:
        return None
    if any(value <= 0 for value in normalized):
        return None
    return normalized


def _format_spacing_text(spacing: tuple[float, float, float] | None) -> str:
    if spacing is None:
        return "暂不可用"
    return " / ".join(f"{value:.3f}" for value in spacing) + " mm"


def _find_first_existing_path(result_dir: Path, filenames: tuple[str, ...]) -> Path | None:
    for filename in filenames:
        candidate = result_dir / filename
        if candidate.is_file():
            return candidate
    return None


def _load_spacing_from_case_meta(case_meta: dict[str, Any] | None) -> tuple[float, float, float] | None:
    return _normalize_spacing((case_meta or {}).get("voxel_spacing"))


def _load_spacing_from_modality(case_meta: dict[str, Any] | None) -> tuple[float, float, float] | None:
    modality_paths = ((case_meta or {}).get("modality_paths") or {})
    for modality in ("flair", "t1ce", "t2", "t1"):
        raw_path = modality_paths.get(modality)
        data, spacing = _load_nifti_data_and_spacing(Path(raw_path)) if raw_path else (None, None)
        if data is not None and spacing is not None:
            return spacing
    return None


def _load_class_mask_volumes(result_dir: Path) -> tuple[dict[str, np.ndarray], tuple[float, float, float] | None, dict[str, str]]:
    mask_volumes: dict[str, np.ndarray] = {}
    spacing: tuple[float, float, float] | None = None
    sources: dict[str, str] = {}

    for class_name, filenames in CLASS_MASK_CANDIDATES.items():
        path = _find_first_existing_path(result_dir, filenames)
        data, current_spacing = _load_nifti_data_and_spacing(path)
        if data is None:
            continue
        mask_volumes[class_name] = np.asarray(data) > 0
        sources[class_name] = path.name if path is not None else ""
        if spacing is None and current_spacing is not None:
            spacing = current_spacing

    return mask_volumes, spacing, sources


def _load_combined_mask_volume(result_dir: Path) -> tuple[np.ndarray | None, tuple[float, float, float] | None, Path | None]:
    path = _find_first_existing_path(result_dir, ("post_combined_label.nii.gz", "combined_label.nii.gz"))
    data, spacing = _load_nifti_data_and_spacing(path)
    if data is None:
        return None, None, None
    return np.asarray(data), spacing, path


def _build_quantitative_analysis(result_dir: Path, case_meta: dict[str, Any] | None) -> dict[str, Any]:
    analysis: dict[str, Any] = {
        "available": False,
        "volume_unit": "ml",
        "wt_volume_ml": None,
        "tc_volume_ml": None,
        "et_volume_ml": None,
        "total_tumor_volume_ml": None,
        "wt_voxels": None,
        "tc_voxels": None,
        "et_voxels": None,
        "total_tumor_voxels": None,
        "spacing": None,
        "spacing_text": "暂不可用",
        "notes": [],
        "primary_note": "当前病例缺少可用于计算的分割结果或体素间距信息。",
        "display_cards": [],
    }

    mask_volumes, mask_spacing, mask_sources = _load_class_mask_volumes(result_dir)
    combined_volume, combined_spacing, combined_path = _load_combined_mask_volume(result_dir)

    if combined_volume is not None:
        combined_masks = {
            "WT": combined_volume > 0,
            "TC": np.isin(combined_volume, (1, 4)),
            "ET": combined_volume == 4,
        }
        for class_name, mask in combined_masks.items():
            if class_name not in mask_volumes:
                mask_volumes[class_name] = mask
                if combined_path is not None:
                    mask_sources[class_name] = combined_path.name

    spacing = (
        _load_spacing_from_case_meta(case_meta)
        or mask_spacing
        or combined_spacing
        or _load_spacing_from_modality(case_meta)
    )
    analysis["spacing"] = list(spacing) if spacing is not None else None
    analysis["spacing_text"] = _format_spacing_text(spacing)

    if spacing is None:
        analysis["notes"].append("缺少体素间距信息，暂无法完成体积计算。")
    if not mask_volumes:
        analysis["notes"].append("当前病例缺少可用于定量分析的分割结果。")

    missing_classes = [class_name for class_name in ("WT", "TC", "ET") if class_name not in mask_volumes]
    if missing_classes:
        analysis["notes"].append(f"部分分区结果暂不可用：{' / '.join(missing_classes)}。")

    if combined_volume is not None and any(source.endswith("combined_label.nii.gz") for source in mask_sources.values()):
        analysis["notes"].append("部分定量结果基于当前合并分割结果推算。")

    if spacing is None or "WT" not in mask_volumes:
        if analysis["notes"]:
            analysis["primary_note"] = analysis["notes"][0]
        return analysis

    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    voxel_counts: dict[str, int | None] = {}
    for class_name in ("WT", "TC", "ET"):
        mask = mask_volumes.get(class_name)
        voxel_counts[class_name] = int(np.count_nonzero(mask)) if mask is not None else None

    total_tumor_voxels = voxel_counts["WT"]
    volume_map = {
        "WT": (voxel_counts["WT"] * voxel_volume_mm3 / 1000.0) if voxel_counts["WT"] is not None else None,
        "TC": (voxel_counts["TC"] * voxel_volume_mm3 / 1000.0) if voxel_counts["TC"] is not None else None,
        "ET": (voxel_counts["ET"] * voxel_volume_mm3 / 1000.0) if voxel_counts["ET"] is not None else None,
        "TOTAL": (total_tumor_voxels * voxel_volume_mm3 / 1000.0) if total_tumor_voxels is not None else None,
    }

    analysis.update(
        {
            "available": True,
            "wt_volume_ml": volume_map["WT"],
            "tc_volume_ml": volume_map["TC"],
            "et_volume_ml": volume_map["ET"],
            "total_tumor_volume_ml": volume_map["TOTAL"],
            "wt_voxels": voxel_counts["WT"],
            "tc_voxels": voxel_counts["TC"],
            "et_voxels": voxel_counts["ET"],
            "total_tumor_voxels": total_tumor_voxels,
            "primary_note": "三维定量结果基于当前分割结果自动计算。",
        }
    )

    pairwise_equal = {
        "WT_TC": bool(
            voxel_counts["WT"] is not None
            and voxel_counts["TC"] is not None
            and np.array_equal(mask_volumes.get("WT"), mask_volumes.get("TC"))
        ),
        "TC_ET": bool(
            voxel_counts["TC"] is not None
            and voxel_counts["ET"] is not None
            and np.array_equal(mask_volumes.get("TC"), mask_volumes.get("ET"))
        ),
        "WT_ET": bool(
            voxel_counts["WT"] is not None
            and voxel_counts["ET"] is not None
            and np.array_equal(mask_volumes.get("WT"), mask_volumes.get("ET"))
        ),
    }
    analysis["pairwise_equal"] = pairwise_equal
    analysis["all_equal"] = bool(all(pairwise_equal.values()))
    analysis["hierarchy_order_valid"] = bool(
        voxel_counts["ET"] is not None
        and voxel_counts["TC"] is not None
        and voxel_counts["WT"] is not None
        and voxel_counts["ET"] <= voxel_counts["TC"] <= voxel_counts["WT"]
    )

    if analysis["all_equal"]:
        analysis["notes"].append("当前分区结果差异较小，建议结合整体病灶结果综合查看。")
    if not analysis["hierarchy_order_valid"]:
        analysis["notes"].append("当前分区体积关系存在异常，建议结合整体病灶结果审慎解读。")

    analysis["display_cards"] = [
        {
            "key": "total",
            "label": "总肿瘤体积",
            "volume_text": (f"{volume_map['TOTAL']:.3f} ml" if volume_map["TOTAL"] is not None else "暂不可用"),
            "voxels_text": (
                f"{_format_int(total_tumor_voxels)} 体素" if total_tumor_voxels is not None else "体素数暂不可用"
            ),
            "emphasis": True,
        },
        {
            "key": "wt",
            "label": "WT 体积",
            "volume_text": (f"{volume_map['WT']:.3f} ml" if volume_map["WT"] is not None else "暂不可用"),
            "voxels_text": (
                f"{_format_int(voxel_counts['WT'])} 体素" if voxel_counts["WT"] is not None else "体素数暂不可用"
            ),
            "emphasis": False,
        },
        {
            "key": "tc",
            "label": "TC 体积",
            "volume_text": (f"{volume_map['TC']:.3f} ml" if volume_map["TC"] is not None else "暂不可用"),
            "voxels_text": (
                f"{_format_int(voxel_counts['TC'])} 体素" if voxel_counts["TC"] is not None else "体素数暂不可用"
            ),
            "emphasis": False,
        },
        {
            "key": "et",
            "label": "ET 体积",
            "volume_text": (f"{volume_map['ET']:.3f} ml" if volume_map["ET"] is not None else "暂不可用"),
            "voxels_text": (
                f"{_format_int(voxel_counts['ET'])} 体素" if voxel_counts["ET"] is not None else "体素数暂不可用"
            ),
            "emphasis": False,
        },
    ]

    return analysis


def _extract_case_metrics(summary_metrics: dict[str, Any] | None, case_id: str) -> dict[str, Any] | None:
    if not summary_metrics:
        return None
    for case_entry in summary_metrics.get("cases", []):
        if case_entry.get("case_id") == case_id:
            return case_entry
    return None


def _build_summary_lines(
    case_meta: dict[str, Any] | None,
    pipeline_summary: dict[str, Any] | None,
    case_metrics: dict[str, Any] | None,
    postprocess_report: dict[str, Any] | None,
    summary_markdown: str | None,
) -> list[str]:
    lines: list[str] = []
    if pipeline_summary:
        if pipeline_summary.get("status") == "completed":
            lines.append("该病例已完成自动处理，可查看分割结果、三维模型与关键切片。")
        elif pipeline_summary.get("status") == "failed":
            lines.append("该病例处理未完成，当前页面展示已生成的结果内容。")

    if postprocess_report:
        class_report = postprocess_report.get("classes", {})
        wt_after = ((class_report.get("WT") or {}).get("after_hierarchy"))
        tc_after = ((class_report.get("TC") or {}).get("after_hierarchy"))
        et_after = ((class_report.get("ET") or {}).get("after_hierarchy"))
        if any(value is not None for value in (et_after, tc_after, wt_after)):
            lines.append("后处理已完成，可继续查看更新后的分割结果与三维模型。")

    if not lines and summary_markdown:
        extracted = []
        for line in summary_markdown.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("|"):
                continue
            if stripped.startswith("- "):
                stripped = stripped[2:]
            extracted.append(stripped)
            if len(extracted) >= 2:
                break
        lines.extend(extracted)

    if not lines:
        lines.append("页面展示病例信息、处理状态、三维结果与关键切片。")
    return lines[:3]


def _normalize_status_detail(detail: Any) -> str:
    text = str(detail or "").strip()
    if not text:
        return "等待处理"

    replacements = (
        ("正在调用 sam_med2d_finetune.inference.volume", "正在进行自动分割"),
        ("正在执行 sam_med2d_finetune.inference.volume", "正在进行自动分割"),
        ("正在调用 infer_volume.py", "正在进行自动分割"),
        ("正在执行 infer_volume.py", "正在进行自动分割"),
        ("正在执行自动分割脚本 infer_volume.py", "正在进行自动分割"),
        ("已生成 ET / TC / WT / combined_label", "分割结果已生成"),
        ("缺少原始分割结果", "暂未生成分割结果"),
        ("正在生成 post_* 结果", "正在进行结果处理"),
        ("已生成 post_* 结果", "后处理完成"),
        ("已完成 3D 后处理与层级约束", "后处理完成"),
        ("已完成后处理并生成 post_* 结果", "后处理完成"),
        ("后处理完成，已生成 post_* 结果", "后处理完成，结果已更新"),
        ("HTML 预览已生成", "三维结果已生成"),
        ("3D HTML 结果已生成", "三维模型已生成"),
        ("HTML 预览已就绪", "三维结果可查看"),
        ("未找到 3D HTML 预览", "暂未生成三维结果"),
        ("正在调用 sam_med2d_finetune.inference.visualize", "正在生成三维结果"),
        ("正在执行 sam_med2d_finetune.inference.visualize", "正在生成三维结果"),
        ("正在调用 visualize_case.py", "正在生成三维结果"),
        ("正在执行 visualize_case.py", "正在生成三维结果"),
        ("正在执行 3D 重建与可视化", "正在生成三维结果"),
        ("3D HTML 预览未生成成功。", "三维结果未生成成功。"),
        ("3D HTML 结果已生成", "三维模型已生成"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _build_quantitative_summary(analysis: dict[str, Any], mode_key: str) -> dict[str, Any]:
    summary = {
        "available": bool(analysis.get("available")),
        "total_tumor_volume_ml": analysis.get("total_tumor_volume_ml"),
        "wt_volume_ml": analysis.get("wt_volume_ml"),
        "tc_volume_ml": analysis.get("tc_volume_ml") if mode_key == "multiclass" else None,
        "et_volume_ml": analysis.get("et_volume_ml") if mode_key == "multiclass" else None,
        "volume_unit": analysis.get("volume_unit", "ml"),
        "cards": list(analysis.get("display_cards") or []),
    }
    return summary


def _build_viewer_controls(result_id: str, mode_key: str) -> dict[str, Any]:
    if mode_key != "multiclass":
        return {
            "enabled": False,
            "strategy": "single-view",
            "mask_urls": {},
            "default_mask": "WT",
        }

    mask_urls = {
        "all": f"/viewer/{result_id}?mode={mode_key}&mask=all",
        "WT": f"/viewer/{result_id}?mode={mode_key}&mask=WT",
        "TC": f"/viewer/{result_id}?mode={mode_key}&mask=TC",
        "ET": f"/viewer/{result_id}?mode={mode_key}&mask=ET",
    }
    return {
        "enabled": True,
        "strategy": "plotly-trace-with-fallback",
        "mask_urls": mask_urls,
        "default_mask": "all",
        "labels": {
            "WT": "\u6574\u4f53\u80bf\u7624\uff08WT\uff09",
            "TC": "\u80bf\u7624\u6838\u5fc3\uff08TC\uff09",
            "ET": "\u589e\u5f3a\u533a\uff08ET\uff09",
        },
    }


def _build_status_cards(
    result_dir: Path,
    pipeline_summary: dict[str, Any] | None,
    *,
    mode_key: str,
) -> list[dict[str, str]]:
    if pipeline_summary and pipeline_summary.get("steps"):
        return [
            {
                "label": str(step.get("name", "未命名步骤")),
                "state": str(step.get("status", "pending")),
                "detail": _normalize_status_detail(step.get("detail", "")),
            }
            for step in pipeline_summary["steps"]
        ]

    raw_ready = all((result_dir / filename).is_file() for filename in RAW_MASK_FILES)
    post_ready = all((result_dir / filename).is_file() for filename in POST_MASK_FILES)
    viewer_mask = DEFAULT_VIEWER_MASK_BY_MODE[normalize_demo_mode(mode_key)]
    viewer_ready = find_viewer_file(result_dir, mode_key, mask_name=viewer_mask) is not None or _can_render_viewer(
        result_dir,
        viewer_mask,
    )
    return [
        {
            "label": "自动分割",
            "state": "done" if raw_ready else "missing",
            "detail": "分割结果已生成" if raw_ready else "暂未生成分割结果",
        },
        {
            "label": "后处理",
            "state": "done" if post_ready else "partial",
            "detail": "后处理完成" if post_ready else "后处理结果不完整",
        },
        {
            "label": "3D 重建",
            "state": "done" if viewer_ready else "missing",
            "detail": "三维结果可查看" if viewer_ready else "暂未生成三维结果",
        },
    ]


def load_result_view(result_id: str, mode_key: str | None = None) -> dict[str, Any]:
    result_dir = decode_result_id(result_id)
    case_meta = _read_json(result_dir / "case_meta.json")
    pipeline_summary = _read_json(_find_nearest_file(result_dir, (PIPELINE_SUMMARY_FILE,)))
    summary_metrics = _read_json(_find_nearest_file(result_dir, SUMMARY_JSON_CANDIDATES))
    summary_markdown = _read_text(_find_nearest_file(result_dir, SUMMARY_MD_CANDIDATES))
    postprocess_report = _read_json(result_dir / "postprocess_report.json")
    resolved_mode_key = resolve_result_mode(
        result_dir,
        requested_mode=mode_key,
        case_meta=case_meta,
        pipeline_summary=pipeline_summary,
    )
    mode = get_demo_mode(resolved_mode_key)

    case_id = (case_meta or {}).get("case_id", result_dir.name)
    case_metrics = _extract_case_metrics(summary_metrics, case_id)
    viewer_mask = DEFAULT_VIEWER_MASK_BY_MODE[resolved_mode_key]
    viewer_file = find_viewer_file(result_dir, resolved_mode_key, mask_name=viewer_mask, auto_generate=True)
    slice_images = _generate_slice_gallery(result_dir, result_id, case_meta)
    analysis = _build_quantitative_analysis(result_dir, case_meta)
    analysis["panel_title"] = str(mode["analysis_title"])
    analysis["panel_description"] = str(mode["analysis_description"])
    analysis["display_cards"] = [
        item for item in analysis["display_cards"] if item["key"] in set(mode["analysis_card_keys"])
    ]
    if bool(mode["show_multiclass_details"]):
        if analysis.get("tc_volume_ml") is None or analysis.get("et_volume_ml") is None:
            analysis["notes"].append("当前病例的部分分区结果暂不可用，页面已优先展示可用结果。")
    else:
        analysis["notes"] = [
            note for note in analysis["notes"]
            if "分区结果差异较小" not in note and "分区体积关系存在异常" not in note
        ]
        if analysis["available"]:
            analysis["notes"].append(str(mode["warning_copy"]))

    if bool(mode["show_multiclass_details"]):
        if analysis.get("tc_volume_ml") is None or analysis.get("et_volume_ml") is None:
            analysis["notes"] = [
                note
                for note in analysis["notes"]
                if note != "\u5f53\u524d\u75c5\u4f8b\u6682\u672a\u5b8c\u6574\u751f\u6210\u6240\u6709\u5206\u533a\u7ed3\u679c\uff0c\u9875\u9762\u5df2\u4f18\u5148\u5c55\u793a\u53ef\u7528\u5185\u5bb9\u3002"
            ]
            analysis["notes"].append(
                "\u5f53\u524d\u75c5\u4f8b\u6682\u672a\u5b8c\u6574\u751f\u6210\u6240\u6709\u5206\u533a\u7ed3\u679c\uff0c\u9875\u9762\u5df2\u4f18\u5148\u5c55\u793a\u53ef\u7528\u5185\u5bb9\u3002"
            )
    else:
        analysis["notes"] = [
            note
            for note in analysis["notes"]
            if note != str(mode["warning_copy"])
            and "鍒嗗尯缁撴灉宸紓杈冨皬" not in note
            and "鍒嗗尯浣撶Н鍏崇郴瀛樺湪寮傚父" not in note
        ]
        analysis["tc_volume_ml"] = None
        analysis["et_volume_ml"] = None
        analysis["tc_voxels"] = None
        analysis["et_voxels"] = None
        analysis["notes"] = [
            note
            for note in analysis["notes"]
            if "TC" not in note and "ET" not in note and "\u5206\u533a" not in note
        ]

    case_info = [
        {"label": "病例编号", "value": str(case_id)},
        {"label": "查看模式", "value": str(mode["label"])},
        {"label": "结果位置", "value": str(result_dir.resolve())},
    ]
    if case_meta and case_meta.get("shape"):
        case_info.append({"label": "体素尺寸", "value": " x ".join(str(item) for item in case_meta["shape"])})
    if case_meta and case_meta.get("voxel_spacing"):
        spacing = " / ".join(str(item) for item in case_meta["voxel_spacing"]) + " mm"
        case_info.append({"label": "体素间距", "value": spacing})
    metric_badges = []

    summary_lines = _build_summary_lines(
        case_meta=case_meta,
        pipeline_summary=pipeline_summary,
        case_metrics=case_metrics,
        postprocess_report=postprocess_report,
        summary_markdown=summary_markdown,
    )
    if bool(mode["show_multiclass_details"]):
        summary_lines = [str(mode["description"]), *summary_lines]
    else:
        summary_lines = [str(mode["description"]), *summary_lines]

    quantitative_summary = _build_quantitative_summary(analysis, resolved_mode_key)
    viewer_controls = _build_viewer_controls(result_id, resolved_mode_key)
    viewer_controls["enabled"] = bool(viewer_file) and bool(viewer_controls["enabled"])

    return {
        "result_id": result_id,
        "case_id": case_id,
        "mode": {
            "key": str(mode["key"]),
            "label": str(mode["label"]),
            "short_label": str(mode["short_label"]),
            "description": str(mode["description"]),
            "show_multiclass_details": bool(mode["show_multiclass_details"]),
        },
        "supports_multiclass": bool(mode["show_multiclass_details"]),
        "viewer_file": viewer_file,
        "viewer_path": str(viewer_file.resolve()) if viewer_file else None,
        "viewer_url": (
            f"/viewer/{result_id}?mode={resolved_mode_key}&mask={viewer_controls['default_mask']}"
            if viewer_file
            else None
        ),
        "viewer_title": str(mode["viewer_title"]),
        "viewer_description": str(mode["viewer_description"]),
        "viewer_controls": viewer_controls,
        "case_info": case_info,
        "status_cards": _build_status_cards(result_dir, pipeline_summary, mode_key=resolved_mode_key),
        "summary_lines": summary_lines[:4],
        "summary_title": str(mode["summary_title"]),
        "summary_description": str(mode["summary_description"]),
        "analysis": analysis,
        "quantitative_summary": quantitative_summary,
        "slice_images": slice_images,
        "slice_title": str(mode["slice_title"]),
        "slice_description": str(mode["slice_description"]),
        "metric_badges": metric_badges,
        "note": str(mode["result_note"]),
    }


def get_viewer_file_for_result(
    result_id: str,
    mode_key: str | None = None,
    mask_name: str | None = None,
) -> Path:
    result_dir = decode_result_id(result_id)
    case_meta = _read_json(result_dir / "case_meta.json")
    pipeline_summary = _read_json(_find_nearest_file(result_dir, (PIPELINE_SUMMARY_FILE,)))
    resolved_mode_key = resolve_result_mode(
        result_dir,
        requested_mode=mode_key,
        case_meta=case_meta,
        pipeline_summary=pipeline_summary,
    )
    viewer_file = find_viewer_file(
        result_dir,
        resolved_mode_key,
        mask_name=mask_name,
        auto_generate=True,
    )
    if viewer_file is None:
        raise FileNotFoundError(f"未找到可显示的三维结果: {result_dir}")
    return viewer_file
