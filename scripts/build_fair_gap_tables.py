#!/usr/bin/env python3
"""Materialize compact paper tables for the corrected fair protocol."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text())


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def add_readouts(rows: list[dict], dataset: str, condition: str, payload: dict) -> None:
    for cell in payload["results"].values():
        rows.append({
            "dataset": dataset, "condition": condition,
            "support_k_per_class": cell["support_k_per_class"],
            "support_k_total": cell["support_k_total"], "method": cell["method"],
            "n_query": cell["n_query"], "accuracy": cell["accuracy"],
            "macro_f1": cell["macro_f1"], "query_label_free": True,
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/fair_gap_tables"))
    args = parser.parse_args()
    root = args.root
    table = []
    add_readouts(table, "MarmAudio", "full_0-8k",
                 load(root / "results/marmaudio_equal_support_readouts_7b_summary.json"))
    add_readouts(table, "BEANS Dogs", "lp_0-1000",
                 load(root / "results/beans_dogs_lp1_equal_support_k1_k2_fullvalid_readouts_7b_summary.json"))
    add_readouts(table, "BEANS Watkins", "full_0-8k",
                 load(root / "results/beans_watkins_equal_support_readouts_7b_summary.json"))
    for dataset, condition, relative in (
        ("MarmAudio", "full_0-8k", "results/marmaudio_equal_support_audio_icl_k1_7b_summary.json"),
        ("MarmAudio", "full_0-8k", "results/marmaudio_equal_support_audio_icl_k2_7b_summary.json"),
        ("MarmAudio", "full_0-8k", "results/marmaudio_equal_support_audio_icl_candidate_k1_7b_summary.json"),
        ("BEANS Dogs", "lp_0-1000", "results/beans_dogs_lp1_equal_support_audio_icl_k1_7b_summary.json"),
    ):
        payload = load(root / relative)
        for cell in payload["results"].values():
            table.append({
                "dataset": dataset, "condition": condition,
                "support_k_per_class": cell["support_k_per_class"],
                "support_k_total": cell["support_k_total"],
                "method": f"audio_icl_{payload['readout']}",
                "n_query": cell["n_query"], "accuracy": cell["accuracy"],
                "macro_f1": cell["macro_f1"], "query_label_free": True,
            })
    write(args.output_dir / "table_equal_supervision.csv", table)

    kv = load(root / "results/beans_dogs_relative_token_kv_lp1_fullvalid_summary.json")
    write(args.output_dir / "table_relative_kv_validation.csv", [
        {**cell, "split": "official_valid", "support_k_per_class": 2,
         "support_k_total": 20, "query_label_free": True}
        for cell in kv["complete_cells"]
    ])
    oracle = load(root / "results/marmaudio_oracle_kv_controls_fullprefill_lp1_7b_summary.json")
    oracle_rows = []
    for method, by_alpha in oracle["cells"].items():
        for alpha, cell in by_alpha.items():
            oracle_rows.append({
                "scope": oracle["scope"], "method": method, "relative_alpha": alpha,
                **cell, "query_label_used": method != "random_field",
            })
    write(args.output_dir / "table_oracle_controls.csv", oracle_rows)

    steering_rows = []
    for relative, dataset in (
        ("results/marmaudio_audio_silence_steering_k1_7b_summary.json", "MarmAudio"),
        ("results/beans_dogs_lp1_audio_silence_steering_icl_options_k1_7b_summary.json",
         "BEANS Dogs A-J"),
    ):
        path = root / relative
        if not path.exists():
            continue
        payload = load(path)
        steering_rows.extend([
            {"dataset": dataset, "state": "baseline", "beta": 0,
             "support_k_total": payload["support_k_total"], "n_query": payload["n_query"],
             **payload["query_baseline"]},
            {"dataset": dataset, "state": "steered", "beta": payload["selected_beta"],
             "support_k_total": payload["support_k_total"], "n_query": payload["n_query"],
             **payload["query_steered"]},
        ])
    if steering_rows:
        write(args.output_dir / "table_audio_silence_steering.csv", steering_rows)

    class_path = root / "results/beans_dogs_probe_routed_class_kv_lp1_valid_summary.json"
    if class_path.exists():
        payload = load(class_path)
        write(args.output_dir / "table_probe_routed_class_kv_validation.csv",
              payload["complete_cells"])
    arbitrary_class_path = root / "results/beans_dogs_probe_routed_class_kv_AJ_lp1_valid_summary.json"
    if arbitrary_class_path.exists():
        payload = load(arbitrary_class_path)
        write(args.output_dir / "table_probe_routed_class_kv_AJ_validation.csv",
              payload["complete_cells"])

    arbitrary_panel_path = root / "results/marmaudio_arbitrary_labels_counterbalanced_summary.json"
    if arbitrary_panel_path.exists():
        payload = load(arbitrary_panel_path)
        write(args.output_dir / "table_marm_arbitrary_label_panel.csv", payload["mappings"])

    beans_zero_path = root / "results/beans_zero_targets_fullscan_cap10_qwen7b_summary.json"
    if beans_zero_path.exists():
        payload = load(beans_zero_path)
        write(args.output_dir / "table_beans_zero_7b.csv", [
            {"component": component, **cell}
            for component, cell in payload.items()
        ])


if __name__ == "__main__":
    main()
