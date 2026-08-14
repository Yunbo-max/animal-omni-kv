#!/usr/bin/env python3
"""Paired uncertainty and exact discordance tests for frozen fair comparisons."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def read(path: Path, predicate=lambda row: True) -> dict[str, dict[str, str]]:
    rows = [row for row in csv.DictReader(path.open(newline="", encoding="utf-8"))
            if predicate(row)]
    result = {row["event_id"]: row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate events after filtering: {path}")
    return result


def paired(
    name: str, a: dict[str, dict[str, str]], b: dict[str, dict[str, str]],
    rng: np.random.Generator, draws: int,
) -> dict:
    if set(a) != set(b):
        raise RuntimeError(f"event mismatch for {name}: {len(a)} vs {len(b)}")
    events = sorted(a)
    a_correct = np.asarray([a[event]["prediction"] == a[event]["target"] for event in events])
    b_correct = np.asarray([b[event]["prediction"] == b[event]["target"] for event in events])
    delta = a_correct.astype(float) - b_correct.astype(float)
    indices = rng.integers(0, len(events), size=(draws, len(events)))
    bootstrap = delta[indices].mean(1)
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    discordant = a_only + b_only
    p_value = float(binomtest(a_only, discordant, .5).pvalue) if discordant else 1.0
    return {
        "comparison": name, "n": len(events),
        "accuracy_a": float(a_correct.mean()), "accuracy_b": float(b_correct.mean()),
        "paired_difference_a_minus_b": float(delta.mean()),
        "bootstrap_95_ci": [float(value) for value in np.quantile(bootstrap, [.025, .975])],
        "a_only_correct": a_only, "b_only_correct": b_only,
        "mcnemar_exact_two_sided_p": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    marm_probe = read(
        root / "results/marmaudio_equal_support_readouts_7b.csv",
        lambda row: row["support_k_per_class"] == "1" and row["method"] == "linear_probe",
    )
    marm_probe_k2 = read(
        root / "results/marmaudio_equal_support_readouts_7b.csv",
        lambda row: row["support_k_per_class"] == "2" and row["method"] == "linear_probe",
    )
    marm_icl_free = read(root / "results/marmaudio_equal_support_audio_icl_k1_7b.csv")
    marm_icl_free_k2 = read(root / "results/marmaudio_equal_support_audio_icl_k2_7b.csv")
    marm_icl_candidate = read(
        root / "results/marmaudio_equal_support_audio_icl_candidate_k1_7b.csv"
    )
    marm_query = set(json.loads(
        (root / "results/marmaudio_equal_support_split_seed20260814.json").read_text()
    )["query_events"])
    marm_candidate_sum = read(
        root / "results/marmaudio_candidate_scoring_full546_7b.csv",
        lambda row: row["event_id"] in marm_query and row["prompt_name"] == "bare",
    )
    for row in marm_candidate_sum.values():
        row["prediction"] = row["prediction_sum"]
    marm_free = read(
        root / "results/expert_qwen7b_b1_clean.csv",
        lambda row: row["event_id"] in marm_query and row["condition"] == "full_0-8k",
    )
    marm_arbitrary_path = root / "results/marmaudio_arbitrary_labels_candidate_k0_k1_7b.csv"
    marm_arbitrary_k0 = read(
        marm_arbitrary_path, lambda row: row["support_k_per_class"] == "0"
    )
    marm_arbitrary_k1 = read(
        marm_arbitrary_path, lambda row: row["support_k_per_class"] == "1"
    )
    for row in list(marm_arbitrary_k0.values()) + list(marm_arbitrary_k1.values()):
        row["target"] = row["target_output"]
    dogs_probe_k1 = read(
        root / "results/beans_dogs_lp1_equal_support_k1_k2_fullvalid_readouts_7b.csv",
        lambda row: row["support_k_per_class"] == "1" and row["method"] == "linear_probe",
    )
    dogs_probe_k2 = read(
        root / "results/beans_dogs_lp1_equal_support_k1_k2_fullvalid_readouts_7b.csv",
        lambda row: row["support_k_per_class"] == "2" and row["method"] == "linear_probe",
    )
    dogs_icl = read(root / "results/beans_dogs_lp1_equal_support_audio_icl_k1_7b.csv")
    dogs_lora = read(root / "results/lora_beans_dogs_lp1_equal_support_k2_7b_valid.csv")
    relative = root / "results/beans_dogs_relative_token_kv_lp1_fullvalid.csv"
    kv_pooled = read(
        relative,
        lambda row: row["method"] == "conditional_pooled" and row["relative_alpha"] == "0.03",
    )
    kv_tokenwise = read(
        relative,
        lambda row: row["method"] == "conditional_tokenwise" and row["relative_alpha"] == "0.03",
    )
    kv_fixed = read(
        relative,
        lambda row: row["method"] == "fixed_mean" and row["relative_alpha"] == "0.03",
    )
    kv_permuted = read(
        relative,
        lambda row: row["method"] == "token_permuted" and row["relative_alpha"] == "0.03",
    )
    rng = np.random.default_rng(args.seed)
    comparisons = [
        paired("Marm K1 probe - zero-shot bare candidate sequence-sum", marm_probe,
               marm_candidate_sum, rng, args.draws),
        paired("Marm bare candidate sequence-sum - free generation", marm_candidate_sum,
               marm_free, rng, args.draws),
        paired("Marm K1 probe - K1 audio ICL free", marm_probe, marm_icl_free, rng, args.draws),
        paired("Marm K1 probe - K1 audio ICL candidate", marm_probe, marm_icl_candidate,
               rng, args.draws),
        paired("Marm K2 probe - K2 audio ICL free", marm_probe_k2, marm_icl_free_k2,
               rng, args.draws),
        paired("Marm arbitrary-label candidate K1 - K0", marm_arbitrary_k1,
               marm_arbitrary_k0, rng, args.draws),
        paired("Dogs K1 probe - K1 audio ICL free", dogs_probe_k1, dogs_icl, rng, args.draws),
        paired("Dogs K2 probe - K2 LoRA", dogs_probe_k2, dogs_lora, rng, args.draws),
        paired("Dogs conditional pooled - fixed mean at alpha=.03", kv_pooled, kv_fixed,
               rng, args.draws),
        paired("Dogs ordered tokenwise - permuted at alpha=.03", kv_tokenwise, kv_permuted,
               rng, args.draws),
        paired("Dogs ordered tokenwise - conditional pooled at alpha=.03", kv_tokenwise,
               kv_pooled, rng, args.draws),
    ]
    payload = {
        "bootstrap_draws": args.draws, "seed": args.seed,
        "inference": "paired example bootstrap; exact two-sided discordant-pair binomial test",
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
