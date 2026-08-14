# Full-extension audit

> This older expansion ledger is paused by the equal-supervision correction in
> `FAIR_GAP_12H_RUN.md`. In particular, the BEANS-Zero scan remains resumable but
> is not running, and no horizontal expansion should supersede the fair-gap and
> validation-gate work summarized in `FAIR_GAP_RESULTS.md`.

This ledger tracks every result previously described as subset, partial, smoke,
single-task, or exploratory. A row is complete only when its authoritative
artifact, protocol, and report integration are all verified.

| Area | Earlier limitation | Required extension | Current evidence | Status |
|---|---|---|---|---|
| AVES-bio / Dogs | CPU extraction stopped after 30/693 | All official train/valid/test representations and fixed-split probe | `results/reps_beans_dogs_aves_bio/` has 693 files; `results/summary_beans_dogs_aves_bio_fixed.json` reports 83.45% test accuracy | Complete; report integration pending |
| AVES-bio / Watkins | CPU extraction stopped after 128/1695 | All official train/valid/test representations and fixed-split probe | `results/reps_beans_watkins_aves_bio/` has 1695 files; `results/summary_beans_watkins_aves_bio_fixed.json` reports 85.84% test accuracy | Complete; report integration pending |
| BEANS-Zero | 25 examples × 12 components | Complete streamed scan of the same 12 requested components under the declared 16 kHz / 10 s cap protocol, then Qwen-7B evaluation | `data/manifests/beans_zero_targets_fullscan_cap10.csv` is resumable and currently growing | Running |
| Thinker LoRA | Watkins only | Add Dogs under the same one-epoch official fixed-split protocol | No authoritative Dogs LoRA artifact yet | Pending |
| Thinker LoRA / MarmAudio | No fixed train/test split; expert corpus only | A same-protocol fixed-test extension is not identifiable without inventing a split; grouped OOF LoRA would answer a different question | Dataset protocol documented in `README.md` | Not applicable; explain in report |
| Conditional KV | MarmAudio, Dogs, and Watkins already evaluated, but Watkins negative | Preserve the negative result; test token-aligned improvement only after validation selection | Absolute-position token-field validation is running; local-token variant implemented and unit-tested | Running |
| Frequency generation | Complete on MarmAudio, Dogs, Watkins | None | Full 11-condition Qwen-7B curves exist for all three | Complete |
| Frequency probe | Full curve only on MarmAudio; Dogs/Watkins full-state probe only | Add degraded fixed-split probe curves if compute remains after higher-priority expansions | No complete cross-frequency Dogs/Watkins probe curves | Pending |
| KV geometry | Main geometry uses 1 kHz degradation | Add cross-frequency gradients/geometry on MarmAudio to test whether the subspace changes with evidence removal | No authoritative lp2/lp4/lp6 geometry yet | Pending |
| Token-preserving KV | Newly motivated by 2026 modality-gap literature | Validation-only comparison against pooled/broadcast KV before untouched test | Support fields complete; absolute-position transfer shows no gain so far; local-state router code passes tests | Running, experimental |
| Smoke/invalid files | Several engineering runs | Keep segregated and excluded from metrics | `results/partial_extensions/` and explicitly named smoke/invalid files | Complete; final audit must verify exclusion |

## Completion gates

1. Complete and evaluate the BEANS-Zero target scan, or record a repeated external
   dataset failure with resumable evidence.
2. Run Dogs Thinker LoRA under the Watkins protocol.
3. Finish the tokenwise validation gate; only touch test if validation improves.
4. Add at least two additional MarmAudio frequency conditions to KV geometry.
5. Decide and document whether Dogs/Watkins degraded probe curves materially alter
   the representation–generation conclusion; run them if feasible.
6. Rebuild tables and figures, update all stale AVES/BEANS-Zero wording, run tests,
   and extend `scripts/audit_artifacts.py` to cover every promoted artifact.
