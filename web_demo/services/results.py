from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import cv2
import nibabel as nib
import numpy as np

from web_demo.config import (
    GENERATED_IMAGE_DIR,
    OUTPUTS_DIR,
    PIPELINE_SUMMARY_FILE,
    POST_MASK_FILES,
    RAW_MASK_FILES,
    SUMMARY_JSON_CANDIDATES,
    SUMMARY_MD_CANDIDATES,
    VIEWER_CANDIDATES,
    ensure_web_demo_dirs,
)


COMBINED_MASK_COLORS = {
    1: np.array([42, 157, 143], dtype=np.uint8),
    2: np.array([69, 123, 157], dtype=np.uint8),
    4: np.array([230, 57, 70], dtype=np.uint8),
}


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


def find_viewer_file(result_dir: Path) -> Path | None:
    for filename in VIEWER_CANDIDATES:
        candidate = result_dir / filename
        if candidate.is_file():
            return candidate
    return None


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
            lines.append("该病例已按单病例串行 demo 流程完成自动分割、后处理和 3D 结果生成。")
        elif pipeline_summary.get("status") == "failed":
            lines.append("该病例运行中断，当前页面只展示已经成功生成的中间结果。")

    if case_metrics and case_metrics.get("post"):
        post_mean_dice = _format_float((case_metrics.get("post") or {}).get("mean_dice"))
        wt_dice = _format_float((((case_metrics.get("post") or {}).get("per_class") or {}).get("WT") or {}).get("dice"))
        if post_mean_dice:
            metric_line = f"已有汇总记录显示该病例后处理后的 mean Dice 为 {post_mean_dice}"
            if wt_dice:
                metric_line += f"，WT Dice 为 {wt_dice}"
            lines.append(metric_line + "。")

    if case_meta and (case_meta.get("prompt_mode") or case_meta.get("finetune_method")):
        config_bits = [bit for bit in (case_meta.get("prompt_mode"), case_meta.get("finetune_method")) if bit]
        lines.append(f"当前结果来自 {' / '.join(config_bits)} 配置下的整病例分割输出。")

    if postprocess_report:
        class_report = postprocess_report.get("classes", {})
        wt_after = ((class_report.get("WT") or {}).get("after_hierarchy"))
        tc_after = ((class_report.get("TC") or {}).get("after_hierarchy"))
        et_after = ((class_report.get("ET") or {}).get("after_hierarchy"))
        if any(value is not None for value in (et_after, tc_after, wt_after)):
            lines.append(
                "后处理结果已落盘，"
                f"ET/TC/WT 的最终体素数分别为 {et_after or 0} / {tc_after or 0} / {wt_after or 0}。"
            )

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
        lines.append("当前页面只保留病例信息、处理状态、3D 预览和 2D 切片，不展示科研看板。")
    return lines[:3]


def _build_status_cards(result_dir: Path, pipeline_summary: dict[str, Any] | None) -> list[dict[str, str]]:
    if pipeline_summary and pipeline_summary.get("steps"):
        return [
            {
                "label": str(step.get("name", "未命名步骤")),
                "state": str(step.get("status", "pending")),
                "detail": str(step.get("detail", "") or "等待执行"),
            }
            for step in pipeline_summary["steps"]
        ]

    raw_ready = all((result_dir / filename).is_file() for filename in RAW_MASK_FILES)
    post_ready = all((result_dir / filename).is_file() for filename in POST_MASK_FILES)
    viewer_ready = find_viewer_file(result_dir) is not None
    return [
        {
            "label": "自动分割",
            "state": "done" if raw_ready else "missing",
            "detail": "已生成 ET / TC / WT / combined_label" if raw_ready else "缺少原始分割结果",
        },
        {
            "label": "后处理",
            "state": "done" if post_ready else "partial",
            "detail": "已生成 post_* 结果" if post_ready else "未找到完整后处理产物",
        },
        {
            "label": "3D 重建",
            "state": "done" if viewer_ready else "missing",
            "detail": "HTML 预览已就绪" if viewer_ready else "未找到 3D HTML 预览",
        },
    ]


def load_result_view(result_id: str) -> dict[str, Any]:
    result_dir = decode_result_id(result_id)
    case_meta = _read_json(result_dir / "case_meta.json")
    pipeline_summary = _read_json(_find_nearest_file(result_dir, (PIPELINE_SUMMARY_FILE,)))
    summary_metrics = _read_json(_find_nearest_file(result_dir, SUMMARY_JSON_CANDIDATES))
    summary_markdown = _read_text(_find_nearest_file(result_dir, SUMMARY_MD_CANDIDATES))
    postprocess_report = _read_json(result_dir / "postprocess_report.json")

    case_id = (case_meta or {}).get("case_id", result_dir.name)
    case_metrics = _extract_case_metrics(summary_metrics, case_id)
    viewer_file = find_viewer_file(result_dir)
    slice_images = _generate_slice_gallery(result_dir, result_id, case_meta)

    case_info = [
        {"label": "病例编号", "value": str(case_id)},
        {"label": "结果目录", "value": str(result_dir.resolve())},
    ]
    if case_meta and case_meta.get("shape"):
        case_info.append({"label": "体素尺寸", "value": " x ".join(str(item) for item in case_meta["shape"])})
    if case_meta and case_meta.get("voxel_spacing"):
        spacing = " / ".join(str(item) for item in case_meta["voxel_spacing"]) + " mm"
        case_info.append({"label": "体素间距", "value": spacing})
    if case_meta and case_meta.get("prompt_mode"):
        case_info.append(
            {
                "label": "分割配置",
                "value": f"{case_meta.get('prompt_mode')} / {case_meta.get('finetune_method', '-')}",
            }
        )

    metric_badges = []
    if case_metrics and case_metrics.get("post"):
        metric_badges.append(
            {"label": "Post Mean Dice", "value": _format_float((case_metrics.get("post") or {}).get("mean_dice")) or "-"}
        )
        metric_badges.append(
            {
                "label": "WT Dice",
                "value": _format_float((((case_metrics.get("post") or {}).get("per_class") or {}).get("WT") or {}).get("dice")) or "-",
            }
        )

    return {
        "result_id": result_id,
        "case_id": case_id,
        "viewer_file": viewer_file,
        "viewer_url": f"/viewer/{result_id}" if viewer_file else None,
        "case_info": case_info,
        "status_cards": _build_status_cards(result_dir, pipeline_summary),
        "summary_lines": _build_summary_lines(
            case_meta=case_meta,
            pipeline_summary=pipeline_summary,
            case_metrics=case_metrics,
            postprocess_report=postprocess_report,
            summary_markdown=summary_markdown,
        ),
        "slice_images": slice_images,
        "metric_badges": metric_badges,
        "note": "页面只保留病例信息、处理状态、3D 结果和 2D 切片，默认不展开科研看板。",
    }


def get_viewer_file_for_result(result_id: str) -> Path:
    result_dir = decode_result_id(result_id)
    viewer_file = find_viewer_file(result_dir)
    if viewer_file is None:
        raise FileNotFoundError(f"未找到 3D HTML 预览: {result_dir}")
    return viewer_file
