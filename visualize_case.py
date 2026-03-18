import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.measure import marching_cubes


MASK_FILES = {
    "ET": "ET.nii.gz",
    "TC": "TC.nii.gz",
    "WT": "WT.nii.gz",
    "combined": "combined_label.nii.gz",
}
POST_PREFIX = "post_"

CLASS_COLORS = {
    "ET": "#e63946",
    "TC": "#2a9d8f",
    "WT": "#457b9d",
}

COMBINED_LABELS = {
    1: ("TC_minus_ET", "#2a9d8f"),
    2: ("WT_minus_TC", "#457b9d"),
    4: ("ET", "#e63946"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a 3D preview for BraTS inference outputs.")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Inference output directory containing case_meta.json and NIfTI masks.",
    )
    parser.add_argument(
        "--mask_name",
        default="all",
        choices=["all", "ET", "TC", "WT", "combined"],
        help="Mask volume to preview. 'all' renders ET/TC/WT together.",
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="Optional HTML output path. Defaults to <output_dir>/preview_3d_<mask_name>.html.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the plotly figure in a browser after saving.",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.45,
        help="Mesh opacity in the 3D preview.",
    )
    return parser.parse_args()


def load_case_meta(output_dir):
    meta_path = Path(output_dir) / "case_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"case_meta.json not found in {output_dir}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_volume(path):
    image = nib.load(str(path))
    return np.asarray(image.dataobj)


def load_postprocess_report(output_dir):
    report_path = Path(output_dir) / "postprocess_report.json"
    if not report_path.is_file():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def resolve_mask_path(output_dir, mask_name, prefix=""):
    output_dir = Path(output_dir)
    filename_prefix = f"{prefix}_" if prefix else ""
    return output_dir / f"{filename_prefix}{MASK_FILES[mask_name]}"


def has_postprocessed_variant(output_dir, mask_name):
    output_dir = Path(output_dir)
    if mask_name == "all":
        required = [output_dir / f"{POST_PREFIX}{MASK_FILES[class_name]}" for class_name in ("ET", "TC", "WT")]
    else:
        required = [resolve_mask_path(output_dir, mask_name, prefix="post")]
    return all(path.is_file() for path in required)


def make_mesh_trace(mask, spacing, name, color, opacity):
    if np.count_nonzero(mask) == 0:
        return None

    padded_mask = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    verts, faces, _, _ = marching_cubes(padded_mask.astype(np.float32), level=0.5, spacing=spacing)
    verts -= np.asarray(spacing, dtype=np.float32)
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        flatshading=True,
        showscale=False,
    )


def build_traces(output_dir, mask_name, spacing, opacity, prefix="", trace_prefix=None):
    output_dir = Path(output_dir)
    traces = []
    trace_prefix = trace_prefix or ""

    if mask_name == "all":
        for class_name in ("ET", "TC", "WT"):
            mask = load_volume(resolve_mask_path(output_dir, class_name, prefix=prefix))
            trace = make_mesh_trace(
                mask=mask > 0,
                spacing=spacing,
                name=f"{trace_prefix}{class_name}",
                color=CLASS_COLORS[class_name],
                opacity=opacity,
            )
            if trace is not None:
                traces.append(trace)
        return traces

    if mask_name == "combined":
        combined = load_volume(resolve_mask_path(output_dir, "combined", prefix=prefix))
        for label_value, (label_name, color) in COMBINED_LABELS.items():
            trace = make_mesh_trace(
                mask=combined == label_value,
                spacing=spacing,
                name=f"{trace_prefix}{label_name}",
                color=color,
                opacity=opacity,
            )
            if trace is not None:
                traces.append(trace)
        return traces

    mask = load_volume(resolve_mask_path(output_dir, mask_name, prefix=prefix))
    trace = make_mesh_trace(
        mask=mask > 0,
        spacing=spacing,
        name=f"{trace_prefix}{mask_name}",
        color=CLASS_COLORS[mask_name],
        opacity=opacity,
    )
    if trace is not None:
        traces.append(trace)
    return traces


def build_report_summary(report):
    if not report:
        return None

    class_summaries = []
    for class_name in ("ET", "TC", "WT"):
        class_report = report.get("classes", {}).get(class_name, {})
        raw_voxels = class_report.get("raw")
        final_voxels = class_report.get("after_hierarchy")
        if raw_voxels is None or final_voxels is None:
            continue
        delta = int(final_voxels) - int(raw_voxels)
        class_summaries.append(f"{class_name} {raw_voxels}\u2192{final_voxels} ({delta:+d})")

    hierarchy_report = report.get("hierarchy", {})
    before_hierarchy = hierarchy_report.get("before", {})
    after_hierarchy = hierarchy_report.get("after", {})
    hierarchy_summary = (
        f"ET\\TC {before_hierarchy.get('et_outside_tc', 0)}\u2192{after_hierarchy.get('et_outside_tc', 0)}, "
        f"TC\\WT {before_hierarchy.get('tc_outside_wt', 0)}\u2192{after_hierarchy.get('tc_outside_wt', 0)}"
    )

    if not class_summaries:
        return hierarchy_summary
    return " | ".join(class_summaries + [hierarchy_summary])


def configure_scene(fig, scene_name):
    fig.update_layout(**{
        scene_name: dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        )
    })


def render_case(output_dir, mask_name="all", save_path=None, show=False, opacity=0.45):
    output_dir = Path(output_dir)
    meta = load_case_meta(output_dir)
    report = load_postprocess_report(output_dir)
    spacing = tuple(float(x) for x in meta.get("voxel_spacing", [1.0, 1.0, 1.0]))
    has_post_variant = has_postprocessed_variant(output_dir, mask_name)

    raw_traces = build_traces(
        output_dir=output_dir,
        mask_name=mask_name,
        spacing=spacing,
        opacity=opacity,
        prefix="",
        trace_prefix="Raw ",
    )
    if not raw_traces:
        raise ValueError(f"No non-empty raw mask surface found for mask_name={mask_name}")

    if has_post_variant:
        post_traces = build_traces(
            output_dir=output_dir,
            mask_name=mask_name,
            spacing=spacing,
            opacity=opacity,
            prefix="post",
            trace_prefix="Post ",
        )
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=("Raw", "Post-processed"),
        )
        for trace in raw_traces:
            fig.add_trace(trace, row=1, col=1)
        for trace in post_traces:
            fig.add_trace(trace, row=1, col=2)
        configure_scene(fig, "scene")
        configure_scene(fig, "scene2")
    else:
        post_traces = []
        fig = go.Figure(data=raw_traces)
        configure_scene(fig, "scene")

    title_text = f"3D preview: {meta['case_id']} ({mask_name})"
    report_summary = build_report_summary(report)
    if has_post_variant and report_summary:
        title_text += f"<br><sup>{report_summary}</sup>"

    fig.update_layout(
        title=title_text,
        margin=dict(l=0, r=0, t=70, b=0),
        legend=dict(x=0.01, y=0.99),
    )

    if save_path is None:
        filename = f"preview_3d_compare_{mask_name}.html" if has_post_variant else f"preview_3d_{mask_name}.html"
        save_path = output_dir / filename
    else:
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path), include_plotlyjs="cdn", full_html=True, auto_open=show)
    return save_path, len(raw_traces) + len(post_traces)


def main():
    args = parse_args()
    save_path, trace_count = render_case(
        output_dir=args.output_dir,
        mask_name=args.mask_name,
        save_path=args.save_path,
        show=args.show,
        opacity=args.opacity,
    )
    print(f"Saved 3D preview to: {save_path} (meshes={trace_count})")


if __name__ == "__main__":
    main()
