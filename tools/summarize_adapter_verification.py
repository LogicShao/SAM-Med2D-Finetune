import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "stage7_adapter_verification"
SUMMARY_DIR = OUTPUT_ROOT / "summary"
TIE_EPS = 1e-8
BOOTSTRAP_SEED = 20260322
BOOTSTRAP_ROUNDS = 10000


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    model: str
    variant: str
    label: str
    summary_json: Path
    result_dir: Path
    case_count: int


RUN_SPECS = (
    RunSpec(
        dataset="fixed20",
        model="adapter",
        variant="baseline",
        label="fixed20 Adapter baseline",
        summary_json=OUTPUT_ROOT / "fixed20_adapter_baseline" / "summary_metrics.json",
        result_dir=OUTPUT_ROOT / "fixed20_adapter_baseline",
        case_count=20,
    ),
    RunSpec(
        dataset="fixed20",
        model="adapter",
        variant="g4",
        label="fixed20 Adapter g4",
        summary_json=OUTPUT_ROOT / "fixed20_adapter_g4" / "summary_metrics.json",
        result_dir=OUTPUT_ROOT / "fixed20_adapter_g4",
        case_count=20,
    ),
    RunSpec(
        dataset="confirm_large_unseen",
        model="adapter",
        variant="baseline",
        label="confirm_large_unseen Adapter baseline",
        summary_json=OUTPUT_ROOT / "confirm_large_unseen_adapter_baseline" / "summary_metrics.json",
        result_dir=OUTPUT_ROOT / "confirm_large_unseen_adapter_baseline",
        case_count=167,
    ),
    RunSpec(
        dataset="confirm_large_unseen",
        model="adapter",
        variant="g4",
        label="confirm_large_unseen Adapter g4",
        summary_json=OUTPUT_ROOT / "confirm_large_unseen_adapter_g4" / "summary_metrics.json",
        result_dir=OUTPUT_ROOT / "confirm_large_unseen_adapter_g4",
        case_count=167,
    ),
    RunSpec(
        dataset="fixed20",
        model="lora",
        variant="baseline",
        label="fixed20 LoRA baseline",
        summary_json=PROJECT_ROOT / "outputs" / "stage1_expand20" / "conf_0p05_iou_0p60" / "summary_metrics.json",
        result_dir=PROJECT_ROOT / "outputs" / "stage1_expand20" / "conf_0p05_iou_0p60",
        case_count=20,
    ),
    RunSpec(
        dataset="fixed20",
        model="lora",
        variant="g4",
        label="fixed20 LoRA g4",
        summary_json=PROJECT_ROOT
        / "outputs"
        / "stage5_expand20_wt_gate_grid"
        / "g4_s008_c72_a033_300_d1_b3"
        / "summary_metrics.json",
        result_dir=PROJECT_ROOT / "outputs" / "stage5_expand20_wt_gate_grid" / "g4_s008_c72_a033_300_d1_b3",
        case_count=20,
    ),
    RunSpec(
        dataset="confirm_large_unseen",
        model="lora",
        variant="baseline",
        label="confirm_large_unseen LoRA baseline",
        summary_json=PROJECT_ROOT
        / "outputs"
        / "stage6_large_confirmation"
        / "runs"
        / "confirm_large_unseen_baseline"
        / "summary_metrics.json",
        result_dir=PROJECT_ROOT / "outputs" / "stage6_large_confirmation" / "runs" / "confirm_large_unseen_baseline",
        case_count=167,
    ),
    RunSpec(
        dataset="confirm_large_unseen",
        model="lora",
        variant="g4",
        label="confirm_large_unseen LoRA g4",
        summary_json=PROJECT_ROOT
        / "outputs"
        / "stage6_large_confirmation"
        / "runs"
        / "confirm_large_unseen_g4_formal"
        / "summary_metrics.json",
        result_dir=PROJECT_ROOT / "outputs" / "stage6_large_confirmation" / "runs" / "confirm_large_unseen_g4_formal",
        case_count=167,
    ),
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_run_metrics(spec: RunSpec) -> dict:
    payload = load_json(spec.summary_json)
    aggregate = payload["aggregate"]["post"]
    wt_continuity = payload.get("aggregate_wt_continuity")
    preview_files = list(spec.result_dir.rglob("preview_3d_compare_all.html"))
    start_ts = spec.result_dir.stat().st_ctime
    end_ts = spec.summary_json.stat().st_mtime
    total_seconds = max(0.0, end_ts - start_ts)

    return {
        "dataset": spec.dataset,
        "model": spec.model,
        "variant": spec.variant,
        "label": spec.label,
        "case_count": spec.case_count,
        "paths": {
            "result_dir": str(spec.result_dir.resolve()),
            "summary_json": str(spec.summary_json.resolve()),
            "summary_md": str((spec.result_dir / "summary.md").resolve()),
            "summary_csv": str((spec.result_dir / "summary_metrics.csv").resolve()),
        },
        "post_mean_dice": float(aggregate["mean_dice"]),
        "post_mean_iou": float(aggregate["mean_iou"]),
        "post_dice": {
            class_name: float(aggregate["per_class"][class_name]["dice"])
            for class_name in ("ET", "TC", "WT")
        },
        "wt_continuity": wt_continuity,
        "timing": {
            "preview_success": len(preview_files),
            "preview_total": spec.case_count,
            "preview_success_rate": float(len(preview_files) / spec.case_count) if spec.case_count else 0.0,
            "total_seconds_estimated": float(total_seconds),
            "avg_seconds_per_case_estimated": float(total_seconds / spec.case_count) if spec.case_count else 0.0,
            "timing_method": "按结果目录创建时间到 summary_metrics.json 修改时间粗略估算",
        },
        "cases": payload["cases"],
    }


def bootstrap_ci(values: list[float], rounds: int = BOOTSTRAP_ROUNDS, seed: int = BOOTSTRAP_SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return [0.0, 0.0]
    samples = rng.choice(array, size=(rounds, array.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return [float(lo), float(hi)]


def build_case_delta_summary(baseline_cases: list[dict], target_cases: list[dict]) -> dict:
    base_map = {item["case_id"]: item for item in baseline_cases}
    target_map = {item["case_id"]: item for item in target_cases}
    deltas = []
    wt_deltas = []
    et_deltas = []
    tc_deltas = []
    detailed = []

    for case_id in sorted(base_map):
        base_item = base_map[case_id]
        target_item = target_map[case_id]
        delta = float(target_item["post"]["mean_dice"] - base_item["post"]["mean_dice"])
        wt_delta = float(
            target_item["post"]["per_class"]["WT"]["dice"] - base_item["post"]["per_class"]["WT"]["dice"]
        )
        et_delta = float(
            target_item["post"]["per_class"]["ET"]["dice"] - base_item["post"]["per_class"]["ET"]["dice"]
        )
        tc_delta = float(
            target_item["post"]["per_class"]["TC"]["dice"] - base_item["post"]["per_class"]["TC"]["dice"]
        )
        deltas.append(delta)
        wt_deltas.append(wt_delta)
        et_deltas.append(et_delta)
        tc_deltas.append(tc_delta)
        detailed.append(
            {
                "case_id": case_id,
                "mean_dice_delta": delta,
                "wt_dice_delta": wt_delta,
                "et_dice_delta": et_delta,
                "tc_dice_delta": tc_delta,
            }
        )

    wins = [item for item in detailed if item["mean_dice_delta"] > TIE_EPS]
    ties = [item for item in detailed if abs(item["mean_dice_delta"]) <= TIE_EPS]
    losses = [item for item in detailed if item["mean_dice_delta"] < -TIE_EPS]
    top_wins = sorted(wins, key=lambda item: item["mean_dice_delta"], reverse=True)[:5]
    top_losses = sorted(losses, key=lambda item: item["mean_dice_delta"])[:5]

    return {
        "win": len(wins),
        "tie": len(ties),
        "loss": len(losses),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(median(deltas)),
        "bootstrap_95_ci_mean_delta": bootstrap_ci(deltas),
        "top5_wins": [[item["case_id"], item["mean_dice_delta"]] for item in top_wins],
        "top5_losses": [[item["case_id"], item["mean_dice_delta"]] for item in top_losses],
        "class_level_case_delta": {
            "mean_wt_delta": float(np.mean(wt_deltas)),
            "median_wt_delta": float(median(wt_deltas)),
            "mean_et_delta": float(np.mean(et_deltas)),
            "median_et_delta": float(median(et_deltas)),
            "mean_tc_delta": float(np.mean(tc_deltas)),
            "median_tc_delta": float(median(tc_deltas)),
        },
    }


def round6(value: float) -> float:
    return float(f"{value:.6f}")


def build_summary_payload() -> dict:
    runs = {}
    for spec in RUN_SPECS:
        run_metrics = build_run_metrics(spec)
        runs[f"{spec.dataset}:{spec.model}:{spec.variant}"] = run_metrics

    datasets = {}
    for dataset in ("fixed20", "confirm_large_unseen"):
        adapter_baseline = runs[f"{dataset}:adapter:baseline"]
        adapter_g4 = runs[f"{dataset}:adapter:g4"]
        lora_baseline = runs[f"{dataset}:lora:baseline"]
        lora_g4 = runs[f"{dataset}:lora:g4"]

        adapter_case_level = build_case_delta_summary(adapter_baseline["cases"], adapter_g4["cases"])
        lora_case_level = build_case_delta_summary(lora_baseline["cases"], lora_g4["cases"])

        datasets[dataset] = {
            "adapter": {
                "baseline": {
                    key: value
                    for key, value in adapter_baseline.items()
                    if key not in {"cases"}
                },
                "g4": {
                    key: value
                    for key, value in adapter_g4.items()
                    if key not in {"cases"}
                },
                "delta_g4_vs_baseline": {
                    "post_mean_dice": float(adapter_g4["post_mean_dice"] - adapter_baseline["post_mean_dice"]),
                    "post_mean_iou": float(adapter_g4["post_mean_iou"] - adapter_baseline["post_mean_iou"]),
                    "post_dice_et": float(
                        adapter_g4["post_dice"]["ET"] - adapter_baseline["post_dice"]["ET"]
                    ),
                    "post_dice_tc": float(
                        adapter_g4["post_dice"]["TC"] - adapter_baseline["post_dice"]["TC"]
                    ),
                    "post_dice_wt": float(
                        adapter_g4["post_dice"]["WT"] - adapter_baseline["post_dice"]["WT"]
                    ),
                },
                "case_level": adapter_case_level,
            },
            "lora": {
                "baseline": {
                    key: value
                    for key, value in lora_baseline.items()
                    if key not in {"cases"}
                },
                "g4": {
                    key: value
                    for key, value in lora_g4.items()
                    if key not in {"cases"}
                },
                "delta_g4_vs_baseline": {
                    "post_mean_dice": float(lora_g4["post_mean_dice"] - lora_baseline["post_mean_dice"]),
                    "post_mean_iou": float(lora_g4["post_mean_iou"] - lora_baseline["post_mean_iou"]),
                    "post_dice_et": float(lora_g4["post_dice"]["ET"] - lora_baseline["post_dice"]["ET"]),
                    "post_dice_tc": float(lora_g4["post_dice"]["TC"] - lora_baseline["post_dice"]["TC"]),
                    "post_dice_wt": float(lora_g4["post_dice"]["WT"] - lora_baseline["post_dice"]["WT"]),
                },
                "case_level": lora_case_level,
            },
            "adapter_vs_lora_delta": {
                "baseline": {
                    "post_mean_dice": float(adapter_baseline["post_mean_dice"] - lora_baseline["post_mean_dice"]),
                    "post_dice_et": float(adapter_baseline["post_dice"]["ET"] - lora_baseline["post_dice"]["ET"]),
                    "post_dice_tc": float(adapter_baseline["post_dice"]["TC"] - lora_baseline["post_dice"]["TC"]),
                    "post_dice_wt": float(adapter_baseline["post_dice"]["WT"] - lora_baseline["post_dice"]["WT"]),
                },
                "g4": {
                    "post_mean_dice": float(adapter_g4["post_mean_dice"] - lora_g4["post_mean_dice"]),
                    "post_dice_et": float(adapter_g4["post_dice"]["ET"] - lora_g4["post_dice"]["ET"]),
                    "post_dice_tc": float(adapter_g4["post_dice"]["TC"] - lora_g4["post_dice"]["TC"]),
                    "post_dice_wt": float(adapter_g4["post_dice"]["WT"] - lora_g4["post_dice"]["WT"]),
                },
            },
        }

    fixed20 = datasets["fixed20"]
    confirm_large_unseen = datasets["confirm_large_unseen"]
    recommendation = {
        "switch_default_baseline_to_adapter": (
            fixed20["adapter_vs_lora_delta"]["baseline"]["post_mean_dice"] > 0
            and confirm_large_unseen["adapter_vs_lora_delta"]["baseline"]["post_mean_dice"] > 0
        ),
        "keep_g4_on_adapter": (
            fixed20["adapter"]["delta_g4_vs_baseline"]["post_mean_dice"] > 0
            and confirm_large_unseen["adapter"]["delta_g4_vs_baseline"]["post_mean_dice"] > 0
        ),
        "web_demo_should_switch_to_adapter_default_model": (
            fixed20["adapter_vs_lora_delta"]["baseline"]["post_mean_dice"] > 0
            and confirm_large_unseen["adapter_vs_lora_delta"]["baseline"]["post_mean_dice"] > 0
        ),
    }
    recommendation["g4_is_stable_across_datasets"] = recommendation["keep_g4_on_adapter"]
    recommendation["report_updates"] = [
        "将 fixed20 与 confirm_large_unseen 的正式 baseline/g4 主结论从 LoRA 口径补齐为 Adapter 口径。",
        "补充 Adapter baseline 与 Adapter g4 的 overall/ET/TC/WT post Dice、case-level win/tie/loss 与 WT continuity 统计。",
        "在默认配置建议中明确：baseline、g4、web demo 默认模型的推荐依据应改为 Adapter 结果，而 LoRA 结果降级为历史对照。",
    ]

    return {
        "meta": {
            "project_root": str(PROJECT_ROOT.resolve()),
            "output_root": str(OUTPUT_ROOT.resolve()),
            "tie_epsilon": TIE_EPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_rounds": BOOTSTRAP_ROUNDS,
            "adapter_checkpoint": str(
                (PROJECT_ROOT / "workdir_multi_task" / "models" / "finetune_adapter" / "best_model.pth").resolve()
            ),
            "lora_checkpoint": str(
                (
                    PROJECT_ROOT
                    / "workdir_multi_task"
                    / "models"
                    / "finetune_no_stop_lora"
                    / "lora_adapters"
                ).resolve()
            ),
        },
        "datasets": datasets,
        "recommendation": recommendation,
    }


def fmt(value: float) -> str:
    return f"{value:.6f}"


def fmt_seconds(value: float) -> str:
    return f"{value:.3f}"


def build_markdown(summary: dict) -> str:
    lines = [
        "# Adapter Verification Comparison",
        "",
        "## 汇总表",
        "",
        "| 数据集 | 模型 | 组别 | Post Mean Dice | ET | TC | WT | 总耗时估算(s) | 平均每例(s) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for dataset in ("fixed20", "confirm_large_unseen"):
        for model in ("adapter", "lora"):
            for variant in ("baseline", "g4"):
                run = summary["datasets"][dataset][model][variant]
                lines.append(
                    f"| {dataset} | {model.upper()} | {variant} | "
                    f"{fmt(run['post_mean_dice'])} | "
                    f"{fmt(run['post_dice']['ET'])} | "
                    f"{fmt(run['post_dice']['TC'])} | "
                    f"{fmt(run['post_dice']['WT'])} | "
                    f"{fmt_seconds(run['timing']['total_seconds_estimated'])} | "
                    f"{fmt_seconds(run['timing']['avg_seconds_per_case_estimated'])} |"
                )

    lines.extend(["", "## 对照结论", ""])

    for dataset in ("fixed20", "confirm_large_unseen"):
        block = summary["datasets"][dataset]
        adapter_delta = block["adapter"]["delta_g4_vs_baseline"]
        adapter_case_level = block["adapter"]["case_level"]
        lora_delta = block["adapter_vs_lora_delta"]
        wt_cont = block["adapter"]["g4"]["wt_continuity"]
        lines.extend(
            [
                f"### {dataset}",
                "",
                f"- Adapter baseline: `post Mean Dice = {fmt(block['adapter']['baseline']['post_mean_dice'])}`，"
                f"`ET = {fmt(block['adapter']['baseline']['post_dice']['ET'])}`，"
                f"`TC = {fmt(block['adapter']['baseline']['post_dice']['TC'])}`，"
                f"`WT = {fmt(block['adapter']['baseline']['post_dice']['WT'])}`",
                f"- Adapter g4: `post Mean Dice = {fmt(block['adapter']['g4']['post_mean_dice'])}`，"
                f"`ET = {fmt(block['adapter']['g4']['post_dice']['ET'])}`，"
                f"`TC = {fmt(block['adapter']['g4']['post_dice']['TC'])}`，"
                f"`WT = {fmt(block['adapter']['g4']['post_dice']['WT'])}`",
                f"- Adapter 内部 g4 相对 baseline: `overall = {fmt(adapter_delta['post_mean_dice'])}`，"
                f"`ET = {fmt(adapter_delta['post_dice_et'])}`，"
                f"`TC = {fmt(adapter_delta['post_dice_tc'])}`，"
                f"`WT = {fmt(adapter_delta['post_dice_wt'])}`",
                f"- LoRA vs Adapter baseline delta: `overall = {fmt(lora_delta['baseline']['post_mean_dice'])}`，"
                f"`ET = {fmt(lora_delta['baseline']['post_dice_et'])}`，"
                f"`TC = {fmt(lora_delta['baseline']['post_dice_tc'])}`，"
                f"`WT = {fmt(lora_delta['baseline']['post_dice_wt'])}`",
                f"- LoRA vs Adapter g4 delta: `overall = {fmt(lora_delta['g4']['post_mean_dice'])}`，"
                f"`ET = {fmt(lora_delta['g4']['post_dice_et'])}`，"
                f"`TC = {fmt(lora_delta['g4']['post_dice_tc'])}`，"
                f"`WT = {fmt(lora_delta['g4']['post_dice_wt'])}`",
                f"- Adapter case-level g4 vs baseline: `win / tie / loss = {adapter_case_level['win']} / {adapter_case_level['tie']} / {adapter_case_level['loss']}`，"
                f"`mean delta = {fmt(adapter_case_level['mean_delta'])}`，"
                f"`bootstrap 95% CI = [{fmt(adapter_case_level['bootstrap_95_ci_mean_delta'][0])}, {fmt(adapter_case_level['bootstrap_95_ci_mean_delta'][1])}]`",
                f"- Adapter g4 WT continuity: `trigger = {wt_cont['trigger_total']}`，"
                f"`rescue = {wt_cont['rescue']}`，`neutral = {wt_cont['neutral']}`，`harm = {wt_cont['harm']}`，"
                f"`reasons = {json.dumps(wt_cont['trigger_reasons'], ensure_ascii=False)}`",
                "",
            ]
        )

    lines.extend(
        [
            "## 建议摘要",
            "",
            f"- 默认 baseline 切换到 Adapter：`{summary['recommendation']['switch_default_baseline_to_adapter']}`",
            f"- Adapter 上保留 g4：`{summary['recommendation']['keep_g4_on_adapter']}`",
            f"- web demo 默认模型切到 Adapter：`{summary['recommendation']['web_demo_should_switch_to_adapter_default_model']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_recommendation_markdown(summary: dict) -> str:
    fixed20 = summary["datasets"]["fixed20"]
    confirm_large_unseen = summary["datasets"]["confirm_large_unseen"]

    switch_default = summary["recommendation"]["switch_default_baseline_to_adapter"]
    keep_g4 = summary["recommendation"]["keep_g4_on_adapter"]
    switch_demo = summary["recommendation"]["web_demo_should_switch_to_adapter_default_model"]

    baseline_sentence = (
        "应从 LoRA baseline 切换到 Adapter baseline。"
        if switch_default
        else "暂不建议从 LoRA baseline 切换到 Adapter baseline。"
    )
    g4_sentence = (
        "Adapter g4 在 fixed20 与 confirm_large_unseen 上都取得正增益，值得保留。"
        if keep_g4
        else "Adapter g4 在至少一个数据集上不稳健，只建议保留为机制验证入口。"
    )
    demo_sentence = (
        "web demo 应切换为 Adapter baseline 默认模型。"
        if switch_demo
        else "web demo 暂不应切换为 Adapter 默认模型。"
    )

    report_lines = "\n".join(f"- {item}" for item in summary["recommendation"]["report_updates"])

    return (
        "# Final Recommendation\n\n"
        "## 直接结论\n\n"
        f"- 默认 baseline 是否应从 LoRA 切换到 Adapter：{baseline_sentence}\n"
        f"- g4 在 Adapter 上是否仍值得保留：{g4_sentence}\n"
        f"- web demo 是否应切换为 Adapter 默认模型：{demo_sentence}\n"
        "- report.md 中哪些结论需要同步更新：\n"
        f"{report_lines}\n\n"
        "## 依据\n\n"
        f"- fixed20：Adapter baseline `post Mean Dice = {fmt(fixed20['adapter']['baseline']['post_mean_dice'])}`，"
        f"LoRA baseline `= {fmt(fixed20['lora']['baseline']['post_mean_dice'])}`，delta `= {fmt(fixed20['adapter_vs_lora_delta']['baseline']['post_mean_dice'])}`。\n"
        f"- fixed20：Adapter g4 `post Mean Dice = {fmt(fixed20['adapter']['g4']['post_mean_dice'])}`，"
        f"相对 Adapter baseline delta `= {fmt(fixed20['adapter']['delta_g4_vs_baseline']['post_mean_dice'])}`。\n"
        f"- confirm_large_unseen：Adapter baseline `post Mean Dice = {fmt(confirm_large_unseen['adapter']['baseline']['post_mean_dice'])}`，"
        f"LoRA baseline `= {fmt(confirm_large_unseen['lora']['baseline']['post_mean_dice'])}`，"
        f"delta `= {fmt(confirm_large_unseen['adapter_vs_lora_delta']['baseline']['post_mean_dice'])}`。\n"
        f"- confirm_large_unseen：Adapter g4 `post Mean Dice = {fmt(confirm_large_unseen['adapter']['g4']['post_mean_dice'])}`，"
        f"相对 Adapter baseline delta `= {fmt(confirm_large_unseen['adapter']['delta_g4_vs_baseline']['post_mean_dice'])}`。\n"
        f"- 若按用户要求的默认切换规则：Adapter baseline 相对 LoRA baseline 在两个数据集上都为正，"
        f"因此 web demo 默认建议切换到 Adapter baseline。\n"
        f"- 若按用户要求的 g4 稳健性规则：Adapter g4 在两个数据集上都为正，"
        f"因此它不属于“两个数据集都不稳健”的情况，但仍应保持为可解释的机制组，不替代 baseline 成为唯一默认。\n"
    )


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary_payload()

    json_path = SUMMARY_DIR / "adapter_comparison.json"
    md_path = SUMMARY_DIR / "adapter_comparison.md"
    recommendation_path = SUMMARY_DIR / "final_recommendation.md"

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(summary), encoding="utf-8")
    recommendation_path.write_text(build_recommendation_markdown(summary), encoding="utf-8")

    print(json.dumps(
        {
            "adapter_comparison_json": str(json_path.resolve()),
            "adapter_comparison_md": str(md_path.resolve()),
            "final_recommendation_md": str(recommendation_path.resolve()),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
