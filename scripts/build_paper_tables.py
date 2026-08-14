#!/usr/bin/env python3
"""Build paper-ready benchmark and adaptation tables from authoritative artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from animal_omni.metrics import classification_metrics


MARM_LABELS = ["Infant Cry", "Phee", "Seep", "Trill", "Tsik", "Twitter"]


def load(path: Path):
    return json.loads(path.read_text())


def condition(summary, name):
    return next(row for row in summary["conditions"] if row["condition"] == name)


def csv_metrics(path: Path, labels: list[str]):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = classification_metrics(
        [row["target"] for row in rows],
        [row["prediction"] or None for row in rows],
        labels,
    )
    return {"n": len(rows), **result}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(); results = root / "results"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    marm3 = condition(load(results / "expert_qwen3b_b1_all_summary.json"), "full_0-8k")
    marm7 = condition(load(results / "expert_qwen7b_b1_summary.json"), "full_0-8k")
    marm_probe = load(results / "probe_7b_oof_summary.json")
    aves_probe = load(results / "aves_bio_probe_oof_summary.json")
    specialist = csv_metrics(results / "expert_marmaudio_specialist.csv", MARM_LABELS)
    dogs_freq = condition(load(results / "beans_dogs_frequency_qwen7b_summary.json"), "full_0-8k")
    dogs_probe = load(results / "beans_dogs_probe_7b_summary.json")
    dogs_aves = load(results / "summary_beans_dogs_aves_bio_fixed.json")
    dogs_kv = load(results / "beans_dogs_conditional_kv_lp1_7b_layer28_rank8_followup_summary.json")
    dogs_cond = next(row for row in dogs_kv["adapted"] if row["method"] == "conditional")
    watkins_freq = condition(load(results / "beans_watkins_frequency_qwen7b_summary.json"), "full_0-8k")
    watkins_probe = load(results / "beans_watkins_probe_7b_summary.json")
    watkins_aves = load(results / "summary_beans_watkins_aves_bio_fixed.json")
    watkins_lora = load(results / "beans_watkins_lora_7b_summary.json")
    watkins_kv = load(results / "beans_watkins_conditional_kv_lp1_7b_summary.json")
    watkins_cond = next(row for row in watkins_kv["adapted"]
                        if row["support_k"] == 20 and row["method"] == "conditional")

    main_rows = [
        {"task": "Call type", "dataset": "MarmAudio", "model_method": "Official ResNet-50",
         "input": "original 0–48 kHz", "protocol": "fixed expert corpus", **specialist},
        {"task": "Call type", "dataset": "MarmAudio", "model_method": "AVES-bio + linear probe",
         "input": "Qwen-matched 0–8 kHz", "protocol": "nested recording-group OOF", "n": aves_probe["n"],
         "accuracy": aves_probe["accuracy"], "macro_f1": aves_probe["macro_f1"]},
        {"task": "Call type", "dataset": "MarmAudio", "model_method": "Qwen-3B zero-shot",
         "input": "observable full 0–8 kHz", "protocol": "fixed expert corpus", "n": marm3["n"],
         "accuracy": marm3["accuracy"], "macro_f1": marm3["macro_f1"]},
        {"task": "Call type", "dataset": "MarmAudio", "model_method": "Qwen-7B zero-shot",
         "input": "observable full 0–8 kHz", "protocol": "fixed expert corpus", "n": marm7["n"],
         "accuracy": marm7["accuracy"], "macro_f1": marm7["macro_f1"]},
        {"task": "Call type", "dataset": "MarmAudio", "model_method": "Qwen-7B + linear probe",
         "input": "observable full 0–8 kHz", "protocol": "nested recording-group OOF", "n": marm_probe["n"],
         "accuracy": marm_probe["accuracy"], "macro_f1": marm_probe["macro_f1"]},
        {"task": "Individual", "dataset": "BEANS Dogs", "model_method": "Qwen-7B zero-shot",
         "input": "observable full 0–8 kHz", "protocol": "official fixed test", "n": dogs_freq["n"],
         "accuracy": dogs_freq["accuracy"], "macro_f1": dogs_freq["macro_f1"]},
        {"task": "Individual", "dataset": "BEANS Dogs", "model_method": "Qwen-7B conditional KV follow-up",
         "input": "low-pass 1 kHz", "protocol": "train support; validation selected; fixed test", "n": dogs_cond["n"],
         "accuracy": dogs_cond["accuracy"], "macro_f1": dogs_cond["macro_f1"]},
        {"task": "Individual", "dataset": "BEANS Dogs", "model_method": "Qwen-7B + linear probe",
         "input": "observable full 0–8 kHz", "protocol": "official fixed test", "n": dogs_probe["n_test"],
         "accuracy": dogs_probe["test_accuracy"], "macro_f1": dogs_probe["test_macro_f1"]},
        {"task": "Individual", "dataset": "BEANS Dogs", "model_method": "AVES-bio + linear probe",
         "input": "Qwen-matched 0–8 kHz", "protocol": "official fixed test", "n": dogs_aves["n_test"],
         "accuracy": dogs_aves["test_accuracy"], "macro_f1": dogs_aves["test_macro_f1"]},
        {"task": "Species", "dataset": "BEANS Watkins", "model_method": "Qwen-7B zero-shot",
         "input": "observable full 0–8 kHz", "protocol": "paired official fixed test", "n": watkins_freq["n"],
         "accuracy": watkins_freq["accuracy"], "macro_f1": watkins_freq["macro_f1"]},
        {"task": "Species", "dataset": "BEANS Watkins", "model_method": "Qwen-7B conditional KV",
         "input": "low-pass 1 kHz", "protocol": "train support; validation selected; fixed test", "n": watkins_cond["n"],
         "accuracy": watkins_cond["accuracy"], "macro_f1": watkins_cond["macro_f1"]},
        {"task": "Species", "dataset": "BEANS Watkins", "model_method": "Qwen-7B Thinker LoRA",
         "input": "observable full 0–8 kHz", "protocol": "one epoch; official fixed test", "n": watkins_lora["n"],
         "accuracy": watkins_lora["accuracy"], "macro_f1": watkins_lora["macro_f1"]},
        {"task": "Species", "dataset": "BEANS Watkins", "model_method": "Qwen-7B + linear probe",
         "input": "observable full 0–8 kHz", "protocol": "official fixed test", "n": watkins_probe["n_test"],
         "accuracy": watkins_probe["test_accuracy"], "macro_f1": watkins_probe["test_macro_f1"]},
        {"task": "Species", "dataset": "BEANS Watkins", "model_method": "AVES-bio + linear probe",
         "input": "Qwen-matched 0–8 kHz", "protocol": "official fixed test", "n": watkins_aves["n_test"],
         "accuracy": watkins_aves["test_accuracy"], "macro_f1": watkins_aves["test_macro_f1"]},
    ]
    write_csv(args.output_dir / "table_main_benchmarks.csv", main_rows)

    marm_cond = load(results / "conditional_kv_lp1k_7b_randomquery_multiseed_summary.json")
    oracle_rows = list(csv.DictReader((results / "oracle_kv_lp1k_7b.csv").open()))
    oracle_columns = [column for column in oracle_rows[0] if column.startswith("correct_eta_")]
    oracle_best = max(sum(row[column] == "true" for row in oracle_rows) / len(oracle_rows)
                      for column in oracle_columns)
    adaptation = [
        {"dataset": "MarmAudio", "protocol": "5 random recording-group query splits; K=20",
         "baseline_accuracy": marm_cond["aggregate"]["baseline_accuracy"]["mean"],
         "fixed_accuracy": marm_cond["aggregate"]["fixed_mean_accuracy"]["mean"],
         "conditional_accuracy": marm_cond["aggregate"]["conditional_accuracy"]["mean"],
         "oracle_recovery": oracle_best, "note": "means over splits; Oracle on eligible failures"},
        {"dataset": "BEANS Dogs", "protocol": "fixed test; K=20 layer28 rank8 follow-up",
         "baseline_accuracy": dogs_kv["baseline"]["accuracy"],
         "fixed_accuracy": next(row for row in dogs_kv["adapted"] if row["method"] == "fixed_mean")["accuracy"],
         "conditional_accuracy": dogs_cond["accuracy"], "oracle_recovery": "",
         "note": "no full-correct/lp1-wrong Oracle-eligible examples"},
        {"dataset": "BEANS Watkins", "protocol": "fixed test; K=20 layer0 rank4",
         "baseline_accuracy": watkins_kv["baseline"]["accuracy"],
         "fixed_accuracy": next(row for row in watkins_kv["adapted"]
                                if row["support_k"] == 20 and row["method"] == "fixed_mean")["accuracy"],
         "conditional_accuracy": watkins_cond["accuracy"], "oracle_recovery": 1.0,
         "note": "Oracle is only 5 eligible examples"},
    ]
    write_csv(args.output_dir / "table_kv_adaptation.csv", adaptation)

    beans_zero = load(results / "beans_zero_core_cap10_qwen7b_summary.json")
    zero_rows = [{"component": key, **value} for key, value in beans_zero.items()]
    write_csv(args.output_dir / "table_beans_zero_core.csv", zero_rows)
    (args.output_dir / "paper_metrics.json").write_text(json.dumps({
        "main_benchmarks": main_rows,
        "kv_adaptation": adaptation,
        "beans_zero_core": zero_rows,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
