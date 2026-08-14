# Equal-supervision and causal-control results

Run window: 2026-08-13 23:50 UTC to 2026-08-14 11:50 UTC.  Model:
Qwen2.5-Omni-7B, Thinker only, BF16, greedy decoding, one RTX 3090.  This report
supersedes every earlier statement that equated zero-shot generation versus a
fully supervised probe with an "Acoustic Grounding Gap."

## Protocol decision

All few-shot comparisons use exactly the same nested K-per-class support IDs.
MarmAudio support/query recordings are disjoint. Dogs and Watkins use official
train support and official validation query. Query labels are used only for
post-hoc scoring. KV interventions are normalized by the relative Frobenius norm
of each modified layer/projection. Validation selection never uses a manifest-
order prefix, and no fair-protocol test output is produced unless the frozen
tokenwise gate passes.

## 1. What candidate scoring changes

On all 546 unanimous-expert MarmAudio events:

| Readout | Accuracy | Macro-F1 |
|---|---:|---:|
| Free generation | 29.49% | 21.94% |
| Bare candidate, mean token log-probability | 30.95% | 20.26% |
| Bare candidate, sequence sum | 33.88% | 25.36% |
| Definition candidate, mean token log-probability | 22.34% | 10.46% |
| Definition candidate, sequence sum | 18.86% | 6.58% |

Bare mean-normalized scoring never predicts Infant Cry, Trill, or Twitter.
Definitions do not repair the label interface. On the frozen balanced 75-query
subset, however, free/mean/sum scores are 41.33/42.67/49.33%. Candidate
normalization therefore explains part of the disparity and must be reported; it
does not support a universal "encoded but cannot decide" claim by itself. In
fact, the K=1 ridge score (46.67%) and zero-shot sequence-sum candidate score
(49.33%) are statistically indistinguishable on these 75 examples (paired
difference -2.67 points, CI -16.00 to +12.00). The robust failure appears when
the native model must consume labeled audio support, or when the frozen readout
gets enough support to rise well above the fixed native candidate boundary.

## 2. Equal-support frozen readouts

### MarmAudio, recording-disjoint query n=75

| K/class | Total support | Audio ICL free | Audio ICL candidate | Centroid | Ridge probe |
|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 8.00% | 4.00% | 29.33% | 46.67% |
| 2 | 12 | 17.33% | not run | 46.67% | 58.67% |
| 4 | 24 | context not run | context not run | 50.67% | 73.33% |
| 8 | 48 | context not run | context not run | 78.67% | 84.00% |

At K=1 the paired probe minus free-ICL difference is +38.67 points (95% paired
bootstrap CI +25.33 to +52.00; exact discordant-pair p=1.08e-6). Against
candidate-scored ICL it is +42.67 points (CI +30.67 to +54.67; p=4.07e-9).
At K=2, free ICL predicts Twitter for all 75 queries. Its 17.33% accuracy and
4.92% macro-F1 are therefore a single-class collapse rather than evidence that
the additional support was used; the same-support ridge result is 58.67%. The
paired ridge-minus-ICL difference is +41.33 points (95% paired bootstrap CI
+28.00--+54.67; exact discordant-pair p=1.23e-7).

Under a frozen bijection from the six calls to arbitrary single-token labels
A--F, zero-support candidate scoring reaches 9.33% and predicts only A/D/B
(48/19/8). One audio demonstration per class changes the support-conditioned
distribution to A/E only (48/27) but reaches just 12.00%. The paired K1-minus-K0
difference is +2.67 points (95% bootstrap CI -2.67--+8.00; exact discordant-pair
p=.625). Thus candidate restriction does not rescue arbitrary label binding;
the demonstrations alter letter priors without a measurable accuracy gain.

A six-mapping cyclic counterbalance then pairs every acoustic class once with
every output letter and moves each letter through every candidate-list position
(450 query--mapping pairs). Averaged over mappings, K0 is 9.56% and K1 is
17.33%; the query-clustered K1--K0 difference is +7.78 points (95% CI
+2.67--+12.44; sign-flip p=.004). This is not evidence of acoustic binding:
K1 is essentially six-way chance and four mappings collapse almost completely
to one letter. A shared-query permutation that preserves every mapping's output
frequency places K1's null interval at 15.33--17.78%; observed 17.33% is not
better than shuffled audio--label association (one-sided p=.152). The support
therefore repairs a bad arbitrary-letter prior toward chance without learning a
reliable query-dependent correspondence.

### BEANS Dogs lp0--1 kHz, official validation n=139

| K/class | Total support | Audio ICL free | Centroid | Ridge probe |
|---:|---:|---:|---:|---:|
| 1 | 10 | 7.19% | 24.46% | 35.25% |
| 2 | 20 | not run | 22.30% | 35.25% |

At K=1 the paired probe minus ICL difference is +28.06 points (CI +20.14 to
+36.69; p=8.65e-10). The result is an arbitrary-category support-utilization
problem, not evidence that Qwen should know dog identity names zero-shot.

### BEANS Watkins full 0--8 kHz, official validation n=339

| K/class | Total support | Centroid | Ridge probe |
|---:|---:|---:|---:|
| 1 | 31 | 26.84% | 32.15% |
| 2 | 62 | 34.22% | 42.48% |
| 4 | 124 | 48.38% | 57.52% |

The old Watkins K=20-total conditional-KV result covered fewer support examples
than labels and is withdrawn from method interpretation.

## 3. Matched-support adaptation on Dogs validation

The exact K=2/class support contains 20 training examples. A one-epoch rank-8
q/v Thinker LoRA predicts Rudy for all 139 validation examples: 2.88% accuracy,
0.56% macro-F1. The same-support ridge probe reaches 35.25%; the paired
difference is +32.37 points (CI +23.74 to +41.01; p=1.97e-11).

Relative-norm KV results on all 139 validation examples:

| Method | alpha=.003 | alpha=.01 | alpha=.03 |
|---|---:|---:|---:|
| Fixed mean | 2.88% | 2.88% | 5.04% |
| Conditional pooled | 2.88% | 2.88% | **12.95%** |
| Conditional ordered tokenwise | 2.88% | 7.19% | 10.79% |
| Token-permuted | 2.88% | 3.60% | 3.60% |
| Matched random | 2.88% | 2.88% | 2.88% |

At alpha=.03 pooled conditional beats fixed by +7.91 points (CI +2.88 to
+13.67; p=.0074). Ordered tokenwise beats its permutation by +7.19 points (CI
+1.44 to +12.95; p=.0309), so token location has a causal effect. But ordered
tokenwise does not beat pooled (-2.16 points, CI -9.35 to +5.04; p=.70), and the
ordering advantage is not stable across alpha. The primary tokenwise validation
gate fails; no fair-protocol test run is permitted.

### Geometry-driven class-dictionary fallback

The preregistered fallback routes each unlabeled validation query with the same
K=2/class ridge readout and selects that pseudo-class's mean support-gradient
field. The router itself is 35.25% accurate. On all 139 validation examples:

| Method | alpha=.003 | alpha=.01 | alpha=.03 |
|---|---:|---:|---:|
| Class-routed pooled | 2.88% | **15.83%** | **20.86%** |
| Class-routed ordered tokenwise | **10.79%** | 13.67% | 0.00% |
| Class-routed token-permuted | 3.60% | 2.16% | 0.72% |

At alpha=.003, ordered tokenwise beats pooled by +7.91 points (95% paired
bootstrap CI +2.88--+13.67; exact discordant-pair p=.0074) and token-permuted
by +7.19 points (CI +2.88--+12.23; p=.0063). This ordering effect is real but
not stable: at alpha=.01 pooled is higher by 2.16 points, and at alpha=.03 the
ordered field produces 99.28% invalid responses. The registered gate required
ordered tokenwise to beat pooled at at least two alpha values. It therefore
fails, and no class-dictionary test evaluation is run. The pooled 20.86%
validation result is a useful partial repair, not permission to promote a
tokenwise method claim.

The registered A--J arbitrary-output replication recomputes all 20 support
gradients against the permuted single-token symbols. Native one-token generation
is 8.63% (macro-F1 2.73%). On the same 139 validation examples:

| Method | alpha=.003 | alpha=.01 | alpha=.03 |
|---|---:|---:|---:|
| Class-routed pooled | 14.39% | 18.71% | **20.14%** |
| Class-routed ordered tokenwise | **15.11%** | 6.47% | 0.00% |
| Class-routed token-permuted | 7.91% | 4.32% | 0.72% |

At alpha=.003 ordered exceeds its token permutation by +7.19 points (paired
bootstrap CI +0.72--+14.39), but the exact discordant-pair test is not
significant at .05 (p=.064). Ordered exceeds pooled by only +0.72 points at this
one alpha (CI -5.76--+7.19; p=1.0), then falls below pooled by 12.23 points at
alpha=.01. At alpha=.03 it produces 99.28% invalid output. Pooled alpha=.03
improves over native by +11.51 points descriptively, but remains below the
35.25% same-support ridge router ceiling. The arbitrary-symbol result therefore
confirms a partial class-conditioned cache effect while independently failing
the tokenwise stability gate; no test evaluation is run.

### Audio--silence steering baseline

A matched conceptual implementation of the 2026 listening intervention averages
the transferred specialist-layer audio-minus-matched-silence residual and adds
it to the final representation. It does not re-discover Omni-specific specialist
heads and is not labeled a faithful reproduction. Support-only beta calibration
selects beta=.1 on MarmAudio and improves the frozen 75-query first-token
candidate readout from 41.33% to 42.67% (+1.33 points). On Dogs, the support
defines arbitrary single-token labels A--J; beta=.3 leaves the complete 139-query
result unchanged at 5.04%. A generic listening direction therefore does not by
itself learn an arbitrary acoustic-category-to-symbol map.

## 4. Oracle capacity and its causal scope

On 12 frozen MarmAudio full-correct/lp1-wrong events:

| Intervention scope/control | Result |
|---|---|
| Full-prefill correct label, alpha=.0003 | 12/12 correct |
| Full-prefill selected wrong label, alpha=.001 | 11/12 hit wrong target; 0/12 correct |
| Full-prefill matched random | 0/12 correct at every registered alpha |
| Full-prefill, audio positions permuted only | 12/12 correct at alpha=.0003 |
| Audio-only correct label, any alpha | 5/12 ever correct |
| Audio-only token permutation, any alpha | 1/12 ever correct |
| Audio-only matched random, any alpha | 0/12 ever correct |

The full-prefill result proves label-specific decision capacity, but its
audio-token permutation control shows that prompt/decision-token gradients may
be decisive. The audio-only result is the narrower evidence for a causal
acoustic-token pathway and is not universal.

## 5. Gradient geometry

Dogs uses exactly two support examples per class. Its best combined K/V layer is
layer 22: same-label cosine 0.862, different-label cosine -0.042, gap 0.903;
global effective rank is 8.04 and between/within trace ratio is 7.68. The support
corrective field is therefore strongly class-conditioned. At layer 22 only about
15.7 of 250 audio tokens carry the effective gradient energy, and 63.95% lies in
the first temporal third.

MarmAudio failure-selected gradients also show strong separation at layer 20
(same 0.838, different -0.190, gap 1.028), but the set contains only four of six
labels and is highly imbalanced. It is mechanism evidence, not a balanced method
training set. Watkins' old 20-gradient set has one example per observed label and
cannot estimate within-class geometry.

## 6. External BEANS-Zero diagnostic

Qwen2.5-Omni-7B Thinker-only is evaluated on all 2,950 examples from the frozen
12-component BEANS-Zero cap-10 manifest using each official instruction,
greedy decoding, a 32-token limit, and canonical exact match:

| Component | n | Exact match |
|---|---:|---:|
| call-type | 283 | **61.48%** |
| zf-indiv | 324 | **66.98%** |
| unseen-family-sci | 130 | 1.54% |
| watkins | 115 | 0.00% |
| unseen-family-cmn / tax | 117 / 123 | 0.00% / 0.00% |
| unseen-genus-cmn / sci / tax | 265 / 262 / 258 | 0.00% / 0.00% / 0.00% |
| unseen-species-cmn / sci / tax | 373 / 364 / 336 | 0.00% / 0.00% / 0.00% |
| **Overall** | **2,950** | **13.32%** |

The 7B model is not uniformly deaf to animal audio: it performs substantially
above chance on the closed song/call and zebra-finch-count tasks. Its native
interface fails almost completely on open taxonomic/species-name binding and on
Watkins species output. This is an external zero-shot diagnostic, not an
equal-support method comparison and not a claim of competition with a
bioacoustically trained model.

As a post-hoc formatting control, a target-aware whole-phrase containment score
is 13.36% overall versus 13.32% exact match. It remains 0% for Watkins and all
open genus/species/common/taxonomic components; unseen-family-sci moves only
from 1.54% exact to 2.31% contained. Thus the near-zero open-label results are
not explained by the model merely wrapping correct names in template sentences.
Containment is diagnostic only and does not replace the registered exact score.

## 7. Invalidated or quarantined artifacts

- The original zero-shot Dogs names versus fully supervised probe comparison is
  an upper-ceiling disparity, not a grounding gap.
- The manifest-order Dogs validation prefix of 64 covered only 6/10 labels and
  is moved to `results/partial_extensions/`.
- Batch=5 changes 2/25 BF16 predictions despite matched norms; all promoted KV
  results use batch=1.
- Raw-eta tokenwise sweeps are engineering pilots. Relative alpha is the only
  promoted intervention scale.
- Oracle 111/111 at some raw eta is replaced by the frozen matched-norm 12-event
  control panel above.
- The old Watkins K=20-total result is not used to infer method failure.

## Current ICLR decision

The equal-support support-to-decision failure is real and statistically strong.
The cache has label-specific causal capacity; support gradients have class and
token structure; and pooled conditional KV yields a real validation gain over a
fixed direction. Neither the continuous tokenwise router nor the preregistered
class-dictionary fallback passes its stability gate. The honest paper is
therefore a diagnostic/causal paper with a partial pooled adaptation result, not
a finished tokenwise-repair methods paper.

The observation that multi-audio demonstrations can hurt Qwen is not itself a
novelty claim: SpeechJudge already reports this behavior for Qwen2.5-Omni-7B,
while Audio Flamingo and SICL-AT show that dedicated interleaved/post-training
can enable useful audio ICL. Context Tuning is also a direct methodological
neighbor because CT-KV optimizes a demonstration-derived KV prefix. The paper's
remaining defensible contribution is the conjunction of equal-supervision
arbitrary bioacoustic binding, causal label/token-resolved gradient geometry,
and a query-conditioned correction placed on the query's own audio state. A
faithful audio CT-KV implementation remains required before a submission can
claim a methods advantage.

Machine-readable evidence: `results/fair_gap_paired_statistics.json`,
`results/fair_gap_artifact_audit.json`, and the per-experiment summaries under
`results/`.
