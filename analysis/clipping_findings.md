# Why the most-clipped saved model looks worst

The saved .01 fits have substantial optimization failures. Their poor ranking does not show that clipping fails to protect extreme predictions. A diagnostic warm-start refit recovers good solutions under exactly the same .01 objective, and one participant gives a particularly clear example of the proposed clipping effect.

## Data and verification

Analyzed all 18 saved CSVs in `delayed_discounting/fit_results`: three discount functions and six choice/scoring rules. Each contains the same 104 participants and 12,271 empirical trials. The analysis uses the current `DelayedDiscountingModels.py` parameterization and the empirical CSV's original within-participant trial order.

Verified that saved trial losses sum to saved NLLs, saved trial probabilities reproduce those losses, BIC matches the parameter count and trial count, and probabilities reconstructed from saved parameters agree with the saved trial losses. The maximum per-trial loss discrepancy is 0.000040, consistent with rounded parameters in the CSVs. Accuracy below is pooled over trials; NLL, BIC, and clipping counts are averages over participants. Lower NLL/BIC is better.

| Family | Saved .01 NLL | Saved .01 BIC | Clips/person | Accuracy | Saved unlimited NLL | Unlimited accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Exponential | 92.67 | 194.88 | 75.98 | 76.78% | 31.54 | 87.94% |
| Hyperbolic | 53.41 | 116.35 | 67.34 | 86.11% | 31.03 | 88.12% |
| Hyperboloid | 43.67 | 101.65 | 72.87 | 86.86% | 28.66 | 89.15% |

## Direct evidence of missed optima

For each participant, evaluate every already-saved parameter vector from the same discount family under the .01 scoring rule. This changes neither the mathematical model, the parameter bounds, nor the scoring rule. A lower score is a direct certificate that the saved .01 fit missed a better feasible solution.

| Family | Saved .01 mean NLL | Best already-saved candidates, rescored at .01 | Participants improved by >0.01 NLL | After diagnostic warm-start refinement |
|---|---:|---:|---:|---:|
| Exponential | 92.67 | 31.34 | 65/104 | 31.15 |
| Hyperbolic | 53.41 | 30.74 | 27/104 | 30.71 |
| Hyperboloid | 43.67 | 28.15 | 48/104 | 28.01 |

The refinement uses L-BFGS-B, with the existing bounds, initialized from the best rescored candidate and the saved unlimited fit. It is diagnostic and does not establish global optimality. Original fits are preserved.

Participant **s519**, exponential: all 120 choices are larger-later. The saved .01 fit has t=6.9132, k=0.9110, predicts the opposite choice on every trial, and has NLL=552.62. The already-saved unlimited parameters t=20, k=0, scored at .01, predict every choice correctly and give NLL=1.206. Both solutions clip on all 120 trials. The clipping count alone cannot distinguish them.

The implementation draws starting parameters uniformly over t in [0.01,20] and k in [0,1] and uses numerical-gradient L-BFGS-B. At many starts the subjective delayed amount is very small and predicted probabilities are far past the clipping boundary. Once both probabilities are clipped, small parameter changes leave the likelihood constant and supply no local gradient. The saved bad parameters, equivalent-count opposite predictions, and recovery using different starts strongly support saturation plateaus as the failure mechanism. The original optimizer traces and random seeds were not saved, so the exact historical trajectory cannot be reconstructed.

Optimization failures also affect other thresholds: exponential e04 improves from mean NLL 147.91 to 31.38 simply by rescoring other saved vectors under e04. The saved threshold sequence is therefore not a reliable comparison of fully optimized models.

## Why frequent clipping does not guarantee a winning score

For an observed choice with assigned probability p, NLL=-ln(p). With symmetric .01 clipping:

- An extreme correct prediction costs -ln(.99)=0.01005.
- An extreme wrong prediction costs -ln(.01)=4.60517.
- Every probability beyond either boundary has the same score as that boundary.

Clipping caps the penalty on extreme mistakes, but the remaining penalty is still substantial. Correct predictions have a best possible loss near zero, and increasing confidence beyond .99 earns no further improvement. Even a model predicting .99 for its selected option on every trial needs about 85.13% accuracy to beat a .5/.5 predictor under this score. This is a comparison against chance, not against the best probabilistic model; calibration and trial difficulty also matter.

In the saved exponential .01 fits, 7,903 trials clip: 6,305 are correct and 1,598 are wrong. Those 1,598 wrong trials contribute 7,359.06 NLL, or 76.36% of the entire model's loss. The 6,305 clipped correct trials contribute just 63.37. The count measures confidence regardless of whether the favored option was observed, so it cannot serve as a performance measure.

## A clear example supporting the original clipping hypothesis

Participant **s524** chose larger-later on 83 of 84 trials. For the exponential model:

| Parameters/scoring | NLL |
|---|---:|
| Saved unlimited fit: t=0.1639, k=0; scored without clipping | 15.30 |
| Extreme feasible solution: t=20, k=0; scored at .01 | 5.44 |
| The same extreme solution: t=20, k=0; scored without clipping, in stable log space | 980.00 |

The extreme solution assigns almost certainty to larger-later on every trial. Its single error has an unclipped loss of approximately 980; clipping replaces this with 4.605. The 83 successes cost 83 times 0.01005 after clipping, for a total of 5.439. Thus the clipped objective prefers a highly overconfident solution to the saved unlimited fit, exactly the proposed mechanism. The original saved .01 fit for this participant had NLL 377.64 and missed that solution entirely.

Across the diagnostic .01 fits, their own mean clipped NLLs are 31.15, 30.71, and 28.01, whereas the same parameter vectors score 56.31, 55.67, and 39.40 without clipping. This demonstrates sensitivity to the scoring rule; comparing different objectives directly does not establish better predictive performance.

## Implementation and interpretation cautions

1. `clipping_count` counts any raw option probability greater than 1-epsilon, irrespective of the observed choice. `trial_prob` instead stores the probability of the observed choice, after clipping. Split clipping into correct and wrong events for interpretation.
2. At epsilon=1e-64, floating-point arithmetic makes 1-epsilon exactly 1.0. The condition `p > 1-epsilon` can never fire. Its saved zero clipping counts do not prove an absence of extreme predictions or lower-tail flooring. For example, 942 trials in the saved exponential e64 fits have a reconstructed option probability below 1e-64, even though the saved clipping count is zero; these are correct extreme predictions, so lower-tail flooring has essentially no effect on their observed-choice loss.
3. `unlimited` explicitly reports zero clipping by definition; this is not a confidence diagnostic. `greedy` is softmax at fixed t=1, not deterministic argmax, and has fewer fitted parameters.
4. Saved BIC values under different clipping rules use different likelihood functions. Interpret each as a score for its particular bounded choice model; demonstrate clipping bias separately by evaluating fixed predictions under multiple floors and by checking model selection under a shared evaluation rule. These are training fits and cannot establish held-out predictive performance or which discount family is true.

Recommended next experiment: retain fixed parameter predictions to isolate pure scoring changes; warm-start each threshold from other good fits, use starting values that cover small k well, and check optimization against feasible saved alternatives. For each family, report clipped-correct count, clipped-wrong count, raw and clipped losses, and selection changes at a common evaluation rule. These changes address the research question more directly than correlating total clipping counts with BIC.

## Reproduce

Run `.venv\Scripts\python.exe analysis\clipping_audit.py` from the project root. The script preserves original results and writes:

- `saved_fit_summary.csv`: aggregate metrics and loss decomposition.
- `saved_subject_diagnostics.csv`: participant-level replay and confidence diagnostics.
- `fixed_parameter_rescoring.csv`: fixed fitted parameters scored at every floor.
- `same_objective_candidate_checks.csv`: better feasible parameter vectors under each saved fit's own objective.
- `warm_start_e02_diagnostic.csv`: diagnostic .01 refinement, parameters, confidence counts, and stable unclipped losses.
