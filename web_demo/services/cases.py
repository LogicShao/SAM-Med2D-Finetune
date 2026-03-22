from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from web_demo.config import SAMPLE_CASE_COLLECTIONS
from web_demo.services.results import encode_result_id, find_viewer_file


@dataclass(frozen=True)
class DemoCase:
    case_id: str
    result_id: str
    result_dir: Path
    source_label: str
    source_tag: str
    summary: str


def list_sample_cases(max_total: int = 6) -> list[DemoCase]:
    demo_cases: list[DemoCase] = []
    seen_case_ids: set[str] = set()

    for collection in SAMPLE_CASE_COLLECTIONS:
        root = Path(collection["root"])
        if not root.is_dir():
            continue

        added = 0
        for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if added >= int(collection["max_cases"]):
                break
            if not (case_dir / "case_meta.json").is_file():
                continue
            if find_viewer_file(case_dir) is None:
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
                    summary="病例处理结果已生成，可直接查看。",
                )
            )
            seen_case_ids.add(case_id)
            added += 1
            if len(demo_cases) >= max_total:
                return demo_cases[:max_total]
    return demo_cases[:max_total]
