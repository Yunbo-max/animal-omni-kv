# Fair-Gap 12-Hour Run

Run window: 2026-08-13 23:50 UTC to 2026-08-14 11:50 UTC.  One RTX 3090
(24 GiB), Qwen2.5-Omni-7B BF16 Thinker-only.  Query labels are unavailable to
all model selection and are used only for final scoring.

## Claim under test

Under identical K-shot-per-class supervision, domain-external acoustic class
information may remain decodable from frozen Qwen states while the model's
native candidate decision remains substantially worse.  A useful cache method
must predict a query-specific corrective field from labeled support, improve an
untouched query without its label, and beat fixed, pooled, permuted-token, and
matched-random interventions at the same relative KV norm.

Zero-shot generation versus a fully supervised probe is retained only as a
descriptive diagnostic and is not called an acoustic-grounding gap.

## Frozen protocol

1. MarmAudio candidate readout: six exact label strings; primary score is mean
   token log probability, secondary is sequence-sum log probability.  Report
   bare-label and acoustic-definition prompts.  Gate on a seeded balanced 120,
   then complete all 546 unanimous-expert examples.
2. Equal supervision: nested K = 1, 2, 4, 8 examples **per class**.  Compare
   audio ICL, cosine nearest centroid, final-layer ridge probe, pooled KV, and
   tokenwise KV using exactly the registered support IDs.
3. MarmAudio query recordings are disjoint from support recordings.  Dogs uses
   the official train/validation/test split.  The first tokenwise gate uses the
   already frozen 20-example Dogs support (exactly two examples per class) and
   the complete official 139-example validation split.  An earlier manifest-
   order prefix of 64 covered only 6/10 classes and is explicitly invalidated.
4. KV scale is relative, not a raw eta.  For every layer/projection and active
   audio positions, `||delta M||_F / ||M||_F = alpha`, where alpha is one of
   0.001, 0.003, 0.01, 0.03, 0.1.
   The complete official validation first evaluates the three middle scales;
   endpoint pilots showed 0.001 is usually inert and 0.1 frequently destroys
   valid output. Endpoints are expanded only if the middle-scale method gate
   passes. A batch=5 acceleration attempt changed 2/25 BF16 decisions despite
   matched norms and is excluded; authoritative runs remain batch=1.
5. Tokenwise validation methods are support mean, query-conditional token
   average, ordered query-conditional token field, seeded token permutation,
   and matched-norm random field.  All modify the same layers and audio-token
   positions.
6. Test is run once only if ordered tokenwise repair beats both conditional
   pooled and fixed mean at the same alpha on validation, and the advantage is
   not isolated to one alpha.

### Registered geometry-driven fallback (before evaluation)

The continuous local-token router failed the primary validation gate.  A single
validation-only fallback is registered from the already observed support
geometry: fit the same-support K=2/class ridge readout, route each unlabeled
query to a pseudo-class, and select that class's mean support gradient field.
Compare pooled, ordered tokenwise, and seeded token-permuted fields at alpha in
{0.003, 0.01, 0.03}.  This fallback passes only if ordered tokenwise exceeds
pooled at at least two alpha values and exceeds token-permuted at the selected
alpha.  Ties select the lower norm.  Otherwise it is reported as a negative
validation result and no fair-protocol test output is generated.

### Registered arbitrary-label replication (before evaluation)

The same Dogs support/query partition is additionally mapped bijectively to
single-token outputs A--J. Support gradients are recomputed against those
arbitrary outputs, not reused from dog-name targets. Parsing accepts only a
label plus optional terminal punctuation; explanations containing incidental
articles ("a") or pronouns ("I") are invalid. Decoding is exactly one new token
because every registered A--J output is a distinct single tokenizer token. The
ridge router still observes
only the same 2/class audio supports and predicts an original acoustic identity;
the selected class field carries the A--J binding. Pooled, ordered, and
token-permuted variants use the same alpha grid and the same frozen gate. This
is a validation replication and does not relax the no-test rule. MarmAudio also
receives K=0 versus K=1/class A--F candidate scoring to distinguish label prior
from demonstration-conditioned arbitrary binding.

After the first A--F mapping showed no significant K1 gain, a robustness panel
was registered before further evaluation: the identity mapping plus all five
cyclic shifts of A--F. Every shift uses the same support/query IDs and candidate
likelihood readout. This exactly counterbalances each acoustic class across all
six output letters; all shifts are reported, with no seed selection.

## Compute order and gates

| Window | Work | Gate / fallback |
|---|---|---|
| 0–1.5 h | MarmAudio candidate scoring; equal-support CPU readouts | If candidate score approaches the full probe, reframe as decoding/formatting and stop KV claims. |
| 1.5–3 h | MarmAudio K=1 audio ICL; Dogs K=2 audio ICL; attempt K=2 MarmAudio if memory permits | Compare only methods using the same registered IDs. OOM falls back to K=1 and records context-length boundary. |
| 3–6 h | Dogs relative-norm tokenwise validation and four matched controls | No test unless the frozen gate above passes. |
| 6–8 h | Correct/wrong/random/shuffled Oracle controls and layerwise class geometry | Oracle is described only as causal capacity. |
| 8–10 h | If token gate passes, one untouched test and MarmAudio transfer. If it fails, label-permutation/equal-support diagnostics replace expansion. | Failure is reported, not tuned on test. |
| 10–12 h | Effective rank, between/within variance, tables, audit, and ICLR claim decision | Every table carries support size, split, supervision, and readout. |

## Decision outcomes

- **Strong path:** equal-support native readout remains below probe; ordered
  tokenwise KV beats pooled/fixed/permuted/random and transfers across prompts.
- **Readout-only path:** candidate scoring or ICL closes the gap; paper becomes
  a decoding/interface study and KV is secondary.
- **Method-negative path:** fair gap remains but conditional tokenwise repair
  fails its gate.  Keep the diagnostic and geometry result, do not claim a
  successful deployable repair method.

## Execution record

The 7B Thinker-only run completed the registered core protocol by 2026-08-14
11:23 UTC on one RTX 3090. Both the original-name and independently recomputed
A--J class-routed tokenwise validations failed the frozen two-alpha gate, so no
test run or cross-prompt promotion was performed. MarmAudio K=2/class free ICL,
the six-mapping counterbalanced A--F candidate panel, and the full 2,950-example
12-component BEANS-Zero 7B screen were completed. The final local audit covers
27 protocol/artifact checks and the test suite contains 19 passing tests. No
GitHub/Hugging Face publication was authorized or performed.
