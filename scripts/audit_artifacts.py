#!/usr/bin/env python3
"""Audit pairing, completeness, split isolation, and deterministic decoding artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def check(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})

    @property
    def passed(self) -> bool:
        return all(item["passed"] for item in self.checks)


def audit_paired(audit: Audit, root: Path, relative: str, expected: set[str]) -> None:
    path = root / relative
    rows = read_csv(path)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["event_id"]].append(row)
    complete = all({row["condition"] for row in items} == expected for items in grouped.values())
    consistent = all(
        len({row["label"] for row in items}) == 1
        and len({row.get("split", "") for row in items}) == 1
        for items in grouped.values()
    )
    files_exist = all(Path(row["audio_path"]).is_file() for row in rows)
    audit.check(f"paired:{relative}", complete and consistent and files_exist,
                f"events={len(grouped)} rows={len(rows)} conditions={len(expected)}")


def audit_generation(audit: Audit, root: Path, predictions: str, manifest: str) -> None:
    prediction_rows = read_csv(root / predictions)
    manifest_rows = read_csv(root / manifest)
    expected = {(row["event_id"], row.get("condition", "full_0-8k")) for row in manifest_rows}
    observed = [(row["event_id"], row["condition"]) for row in prediction_rows]
    unique = len(observed) == len(set(observed))
    batch_one = all(row.get("batch_size", "1") == "1" for row in prediction_rows)
    audit.check(f"generation:{predictions}", unique and batch_one and set(observed) == expected,
                f"expected={len(expected)} observed={len(observed)} unique={unique} batch1={batch_one}")


def audit_fixed_kv(
    audit: Audit,
    root: Path,
    manifest: str,
    support_split: str,
    predictions: str,
) -> None:
    manifest_rows = read_csv(root / manifest)
    split = json.loads((root / support_split).read_text())
    prediction_rows = read_csv(root / predictions)
    by_event = {row["event_id"]: row for row in manifest_rows if row["condition"] == "lp_0-1000"}
    support = set(split["support_order"])
    support_train = all(by_event[event_id]["split"] == "train" for event_id in support)
    query = {row["event_id"] for row in prediction_rows}
    query_test = all(by_event[event_id]["split"] == "test" for event_id in query)
    identities = Counter(
        (row["event_id"], int(row["support_k"]), row["method"])
        for row in prediction_rows
    )
    expected = {
        (event_id, support_k, method)
        for event_id in query
        for support_k in (1, 5, 10, 20)
        for method in ("fixed_mean", "conditional")
    }
    complete = set(identities) == expected and all(value == 1 for value in identities.values())
    audit.check(f"fixed_kv:{predictions}",
                support_train and query_test and support.isdisjoint(query) and complete,
                f"support={len(support)} query={len(query)} train={support_train} "
                f"test={query_test} disjoint={support.isdisjoint(query)} complete={complete}")


def audit_marm_splits(audit: Audit, root: Path) -> None:
    manifest = read_csv(root / "data/manifests/marmaudio_expert_validation_interventions.csv")
    event_group = {row["event_id"]: row["recording_id"] for row in manifest}
    paths = sorted((root / "results").glob("conditional_kv_lp1k_7b_randomquery_seed*_split.json"))
    disjoint = True
    for path in paths:
        split = json.loads(path.read_text())
        query_groups = {event_group[event_id] for event_id in split["query_events"]}
        support_groups = {event_group[event_id] for event_id in split["support_order"]}
        disjoint &= query_groups.isdisjoint(support_groups)
    audit.check("marm_conditional_recording_disjoint", bool(paths) and disjoint,
                f"splits={len(paths)} disjoint={disjoint}")


def audit_simple_counts(audit: Audit, root: Path) -> None:
    expected = {
        "results/beans_zero_core_cap10_qwen7b.csv": (300, ("event_id",)),
        "results/marmaudio_controls_qwen7b.csv": (1800, ("event_id", "condition")),
        "results/marmaudio_prompt_reversed_qwen7b.csv": (300, ("event_id",)),
        "results/marmaudio_prompt_permuted_qwen7b.csv": (300, ("event_id",)),
        "results/beans_dogs_conditional_kv_lp1_7b_layer28_rank8_followup.csv":
            (278, ("event_id", "support_k", "method")),
    }
    for relative, (count, columns) in expected.items():
        rows = read_csv(root / relative)
        identities = [tuple(row[column] for column in columns) for row in rows]
        passed = len(rows) == count and len(identities) == len(set(identities))
        audit.check(f"count:{relative}", passed,
                    f"expected={count} observed={len(rows)} unique={len(set(identities))}")
    summaries = {
        "results/probe_7b_oof_summary.json": 546,
        "results/probe_lp1k_7b_oof_summary.json": 546,
        "results/aves_bio_probe_oof_summary.json": 546,
        "results/beans_dogs_probe_7b_summary.json": 139,
        "results/beans_watkins_probe_7b_summary.json": 339,
    }
    for relative, expected_n in summaries.items():
        value = json.loads((root / relative).read_text())
        observed = value.get("n", value.get("n_test"))
        audit.check(f"summary:{relative}", observed == expected_n,
                    f"expected={expected_n} observed={observed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    audit = Audit()
    all_conditions = {
        "full_0-8k", "lp_0-1000", "lp_0-2000", "lp_0-4000", "lp_0-6000", "lp_0-8000",
        "remove_0-1000", "remove_1000-2000", "remove_2000-4000",
        "remove_4000-6000", "remove_6000-8000",
    }
    for manifest in (
        "data/manifests/marmaudio_expert_validation_interventions.csv",
        "data/manifests/beans_dogs_test_all_interventions.csv",
        "data/manifests/beans_watkins_test_all_interventions.csv",
    ):
        audit_paired(audit, root, manifest, all_conditions)
    generation_pairs = [
        ("results/expert_qwen7b_b1_clean.csv", "data/manifests/marmaudio_expert_validation_interventions.csv"),
        ("results/beans_dogs_frequency_qwen7b.csv", "data/manifests/beans_dogs_test_all_interventions.csv"),
        ("results/beans_watkins_frequency_qwen7b.csv", "data/manifests/beans_watkins_test_all_interventions.csv"),
    ]
    if (root / "results/expert_qwen3b_b1_all.csv").exists():
        generation_pairs.append(("results/expert_qwen3b_b1_all.csv",
                                 "data/manifests/marmaudio_expert_validation_interventions.csv"))
    for predictions, manifest in generation_pairs:
        if (root / predictions).exists():
            audit_generation(audit, root, predictions, manifest)
        else:
            audit.check(f"generation:{predictions}", False, "missing")
    for dataset in ("dogs", "watkins"):
        audit_fixed_kv(
            audit, root,
            f"data/manifests/beans_{dataset}_all_full_lp1.csv",
            f"results/beans_{dataset}_support_k_split.json",
            f"results/beans_{dataset}_conditional_kv_lp1_7b.csv",
        )
    audit_marm_splits(audit, root)
    audit_simple_counts(audit, root)
    result = {"passed": audit.passed, "checks": audit.checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not audit.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
