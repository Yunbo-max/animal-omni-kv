#!/usr/bin/env python3
"""Validation-only class-dictionary KV repair routed by a same-support probe."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from animal_omni.conditional_kv import place_audio_token_delta
from animal_omni.kv_hooks import KVDeltaHooks
from animal_omni.metrics import normalize_label
from animal_omni.qwen_runner import QwenThinkerRunner


def load_representation(root: Path, event_id: str, layer: int) -> np.ndarray:
    with np.load(root / f"{event_id}.npz", allow_pickle=False) as record:
        return record["representation"][layer].astype(np.float32)


def mean_fields(fields: list[dict]) -> dict:
    return {
        key: torch.stack([field[key].float() for field in fields]).mean(0)
        for key in sorted(fields[0])
    }


def token_average(field: dict) -> dict:
    return {
        key: value.mean(0, keepdim=True).expand_as(value).clone()
        for key, value in field.items()
    }


def token_permutation(field: dict, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(next(iter(field.values()))), generator=generator)
    return {key: value[order].clone() for key, value in field.items()}


def stable_seed(base: int, event_id: str) -> int:
    digest = hashlib.sha256(f"{base}:{event_id}:class-permutation".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2 ** 63 - 1)


def atomic_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_prediction(raw: str, labels: list[str], *, strict: bool) -> str:
    if not strict:
        return normalize_label(raw, labels) or ""
    # Arbitrary single-token labels such as A--J must not be recovered from an
    # explanation (where articles like "a" and pronouns like "I" are common).
    # The prompt requests label-only output, so allow only surrounding
    # whitespace and terminal punctuation.
    cleaned = re.sub(r"[\s.,:;!?]+", "", raw)
    return cleaned if cleaned in labels else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--condition", default="lp_0-1000")
    parser.add_argument("--query-split", default="valid")
    parser.add_argument("--equal-support-split", type=Path, required=True)
    parser.add_argument("--support-k-per-class", type=int, default=2)
    parser.add_argument("--gradient-dir", type=Path, required=True)
    parser.add_argument("--representation-dir", type=Path, required=True)
    parser.add_argument("--feature-layer", type=int, default=28)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--label-map", type=Path,
        help="JSON map from dataset labels to arbitrary output strings",
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt", help="override the configured query prompt")
    parser.add_argument("--prompt-name", default="registered")
    parser.add_argument(
        "--max-new-tokens", type=int,
        help="override generation length (use 1 for registered single-token labels)",
    )
    parser.add_argument("--relative-alphas", type=float, nargs="+",
                        default=[.003, .01, .03])
    parser.add_argument("--methods", nargs="+", default=[
        "probe_class_pooled", "probe_class_tokenwise", "probe_class_permuted",
    ])
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    config = yaml.safe_load(args.config.read_text())
    labels = config["dataset"]["labels"]
    label_map = {label: label for label in labels}
    prompt = args.prompt or config["evaluation"]["prompt"]
    if args.label_map:
        label_map = json.loads(args.label_map.read_text())
        if set(label_map) != set(labels) or len(set(label_map.values())) != len(labels):
            raise ValueError("label map must bijectively cover every configured label")
        prompt = args.prompt or (
            "Classify this sound into one of the registered arbitrary acoustic categories. "
            f"Choose exactly one label from: {', '.join(label_map[label] for label in labels)}. "
            "Answer with only the label."
        )
    output_labels = [label_map[label] for label in labels]
    rows = [
        row for row in csv.DictReader(args.manifest.open(newline="", encoding="utf-8"))
        if row["condition"] == args.condition and row["split"] == args.query_split
    ]
    by_event = {
        row["event_id"]: row
        for row in csv.DictReader(args.manifest.open(newline="", encoding="utf-8"))
        if row["condition"] == args.condition
    }
    split = json.loads(args.equal_support_split.read_text())
    support = split["support_sets"][str(args.support_k_per_class)]
    support_y = np.asarray([by_event[event]["label"] for event in support])
    support_x = np.stack([
        load_representation(args.representation_dir, event, args.feature_layer)
        for event in support
    ])
    router = make_pipeline(
        StandardScaler(), RidgeClassifier(alpha=args.ridge_alpha)
    ).fit(support_x, support_y)
    gradients_by_label: dict[str, list[dict]] = defaultdict(list)
    for event, label in zip(support, support_y):
        record = torch.load(
            args.gradient_dir / f"{event}.pt", map_location="cpu", weights_only=True
        )
        gradients_by_label[str(label)].append(record["tokenwise_audio_gradient"])
    if set(gradients_by_label) != set(labels):
        raise RuntimeError("support does not cover every registered label")
    class_fields = {
        label: mean_fields(gradients_by_label[label]) for label in labels
    }

    output = list(csv.DictReader(args.output.open(newline="", encoding="utf-8"))) \
        if args.resume and args.output.exists() else []
    complete = {
        (row["event_id"], row["method"], float(row["relative_alpha"]))
        for row in output
    }
    runner = QwenThinkerRunner(args.model_id)
    audio_token_id = runner.model.thinker.config.audio_token_id
    expected = len(rows) * len(args.methods) * len(args.relative_alphas)
    known_methods = {
        "native", "probe_class_pooled", "probe_class_tokenwise",
        "probe_class_permuted",
    }
    if set(args.methods) - known_methods:
        raise ValueError(f"unknown methods: {sorted(set(args.methods) - known_methods)}")
    if "native" in args.methods and args.relative_alphas != [0.0]:
        raise ValueError("native must be run separately with --relative-alphas 0")

    for row_index, row in enumerate(rows, 1):
        query_x = load_representation(
            args.representation_dir, row["event_id"], args.feature_layer
        )[None, :]
        routed_label = str(router.predict(query_x)[0])
        ordered = class_fields[routed_label]
        fields = {
            "native": None,
            "probe_class_pooled": token_average(ordered),
            "probe_class_tokenwise": ordered,
            "probe_class_permuted": token_permutation(
                ordered, stable_seed(args.seed, row["event_id"])
            ),
        }
        inputs = runner.prepare_inputs(row["audio_path"], prompt)
        input_length = inputs["input_ids"].shape[1]
        audio_mask = inputs["input_ids"].eq(audio_token_id)
        for method in args.methods:
            unit_delta = (
                {} if method == "native"
                else place_audio_token_delta(fields[method], audio_mask, 1.0)
            )
            for relative_alpha in args.relative_alphas:
                identity = (row["event_id"], method, relative_alpha)
                if identity in complete:
                    continue
                hooks = KVDeltaHooks(
                    runner.model.thinker, unit_delta,
                    relative_alpha=None if method == "native" else relative_alpha,
                )
                with torch.inference_mode(), hooks:
                    generated = runner.model.generate(
                        **inputs, return_audio=False, do_sample=False,
                        max_new_tokens=(args.max_new_tokens or
                                        config["evaluation"]["max_new_tokens"]),
                        use_audio_in_video=False,
                    )
                raw = runner.processor.batch_decode(
                    generated[:, input_length:], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                prediction = parse_prediction(
                    raw, output_labels, strict=args.label_map is not None
                )
                ratios = list(hooks.applied_relative_norms.values()) or [0.0]
                output.append({
                    "event_id": row["event_id"], "target": row["label"],
                    "target_output": label_map[row["label"]],
                    "prompt_name": args.prompt_name,
                    "query_split": args.query_split, "condition": args.condition,
                    "support_k_per_class": args.support_k_per_class,
                    "support_k_total": len(support),
                    "router": "same-support standardized ridge classifier",
                    "routed_label": routed_label,
                    "router_correct": str(routed_label == row["label"]).lower(),
                    "method": method, "relative_alpha": relative_alpha,
                    "feature_layer": args.feature_layer,
                    "applied_ratio_mean": float(np.mean(ratios)),
                    "applied_ratio_min": float(np.min(ratios)),
                    "applied_ratio_max": float(np.max(ratios)),
                    "prediction": prediction, "raw_prediction": raw,
                    "correct": str(prediction == label_map[row["label"]]).lower(),
                    "prediction_matches_routed_label": str(
                        prediction == label_map[routed_label]
                    ).lower(),
                })
                complete.add(identity)
                atomic_write(args.output, output)
        if row_index == 1 or row_index % 10 == 0 or len(output) == expected:
            print(f"[{len(output)}/{expected}] {row['event_id']} routed={routed_label}", flush=True)


if __name__ == "__main__":
    main()
