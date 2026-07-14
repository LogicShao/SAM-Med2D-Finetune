from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sam_med2d_finetune.web_demo.config import get_sample_result_collections, normalize_demo_mode
from sam_med2d_finetune.web_demo.services.results import encode_result_id


@dataclass(frozen=True)
class DemoCase:
    case_id: str
    result_id: str
    result_dir: Path
    source_label: str
    source_tag: str
    mode_key: str
    summary: str


def _is_readable_case_result(case_dir: Path) -> bool:
    return (case_dir / "case_meta.json").is_file() and any(
        (case_dir / filename).is_file()
        for filename in ("post_combined_label.nii.gz", "combined_label.nii.gz", "post_WT.nii.gz", "WT.nii.gz")
    )


def list_sample_cases(mode_key: str | None, max_total: int = 6) -> list[DemoCase]:
    selected_mode = normalize_demo_mode(mode_key)
    demo_cases: list[DemoCase] = []
    seen_case_ids: set[str] = set()

    for collection in get_sample_result_collections(selected_mode):
        root = Path(collection["root"])
        if not root.is_dir():
            continue

        added = 0
        for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if added >= int(collection["max_cases"]):
                break
            if not _is_readable_case_result(case_dir):
                continue

            case_id = case_dir.name
            if case_id in seen_case_ids:
                continue

            demo_cases.append(
                DemoCase(
                    case_id=case_id,
                    result_id=encode_result_id(case_dir),
                    result_dir=case_dir,
                    source_label=str(collection["label"]),
                    source_tag=str(collection["tag"]),
                    mode_key=selected_mode,
                    summary=(
                        "\u8be5\u75c5\u4f8b\u5df2\u751f\u6210\u5f53\u524d\u6a21\u5f0f\u7ed3\u679c\uff0c\u53ef\u76f4\u63a5\u67e5\u770b\u3002"
                        if selected_mode == "standard"
                        else "\u8be5\u75c5\u4f8b\u5df2\u751f\u6210\u591a\u7c7b\u522b\u5206\u6790\u7ed3\u679c\uff0c\u53ef\u67e5\u770b WT / TC / ET \u533a\u57df\u5206\u5e03\u3002"
                    ),
                )
            )
            seen_case_ids.add(case_id)
            added += 1
            if len(demo_cases) >= max_total:
                return demo_cases[:max_total]
    return demo_cases[:max_total]
