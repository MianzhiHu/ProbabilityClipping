"""Audit saved fits without changing them. Run from the repository root."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis"
FLOORS = {"e02": .01, "e04": 1e-4, "e16": 1e-16, "e64": 1e-64, "unlimited": 0., "greedy": 0.}
data = pd.read_csv(ROOT / "delayed_discounting/empirical_data/dd_empirical_data.csv")
data["y"] = data.choice.map({"smaller_sooner": 0, "larger_later": 1})
groups = dict(tuple(data.groupby("worker_id", sort=True)))
fits = {}
for path in sorted((ROOT / "delayed_discounting/fit_results").glob("*.csv")):
    family, rule = path.stem.rsplit("_", 1)
    frame = pd.read_csv(path).set_index("participant_id")
    assert frame.index.is_unique and set(frame.index) == set(groups)
    frame["params"] = frame.best_parameters.map(lambda s: np.fromstring(s.strip("[]"), sep=" "))
    fits[family, rule] = frame


def raw_losses(family, params, g, greedy=False):
    t, k = (1., params[0]) if greedy else params[:2]
    a, d = g.large_amount.to_numpy(), g.later_delay.to_numpy()
    if family == "exponential":
        v = a * np.exp(-k*d)
    elif family == "hyperbolic":
        v = a / (1+k*d)
    else:
        s = params[1] if greedy else params[2]
        v = a / (1+k*d)**s
    z = t * (v-g.small_amount.to_numpy())
    # Exact binary log-softmax in log space; no probability underflow.
    loss = np.logaddexp(0., np.where(g.y.to_numpy() == 1, -z, z))
    return loss, z


def score(loss, epsilon):
    return np.clip(loss, -np.log1p(-epsilon), -np.log(epsilon)) if epsilon else loss


summaries, subjects, transfers = [], [], []
max_replay_error = 0.
for (family, rule), f in fits.items():
    eps = FLOORS[rule]
    accum = dict(n=0, correct=0, extreme_correct=0, extreme_wrong=0,
                 extreme_correct_loss=0., extreme_wrong_loss=0., other_loss=0.,
                 score_gain_wrong=0., score_cost_correct=0., raw_nll=0.)
    for sid, r in f.iterrows():
        g = groups[sid]
        loss, z = raw_losses(family, r.params, g, rule == "greedy")
        replay = score(loss, eps)
        saved = np.array(ast.literal_eval(r.trial_nll))
        p = np.array(ast.literal_eval(r.trial_prob))
        assert len(saved) == len(p) == len(g)
        assert np.allclose(saved, -np.log(p), atol=1e-10, rtol=1e-10)
        assert np.isclose(saved.sum(), r.best_nll)
        assert np.allclose(saved, replay, atol=1e-3, rtol=1e-5)
        assert np.isclose(r.BIC, len(r.params)*np.log(len(g)) + 2*r.best_nll)
        max_replay_error = max(max_replay_error, float(np.max(np.abs(saved-replay))))
        threshold = np.log((1-eps)/eps) if eps else np.inf
        extreme = np.abs(z) > threshold
        correct = (z > 0) == g.y.to_numpy()  # Match argmax: exact ties select option 0.
        ec, ew = extreme & correct, extreme & ~correct
        accum["n"] += len(g)
        accum["correct"] += np.sum(correct)
        accum["extreme_correct"] += np.sum(ec)
        accum["extreme_wrong"] += np.sum(ew)
        accum["extreme_correct_loss"] += saved[ec].sum()
        accum["extreme_wrong_loss"] += saved[ew].sum()
        accum["other_loss"] += saved[~extreme].sum()
        accum["score_gain_wrong"] += np.maximum(loss-replay, 0).sum()
        accum["score_cost_correct"] += np.maximum(replay-loss, 0).sum()
        accum["raw_nll"] += loss.sum()
        subjects.append(dict(family=family, rule=rule, participant_id=sid,
                             saved_nll=r.best_nll, replay_nll=replay.sum(), raw_nll=loss.sum(),
                             n=len(g), accuracy=correct.mean(), saved_clipping_count=r.clipping_count,
                             extreme_correct=ec.sum(), extreme_wrong=ew.sum()))
    summaries.append(dict(family=family, rule=rule, subjects=len(f),
                          mean_nll=f.best_nll.mean(), mean_bic=f.BIC.mean(),
                          mean_saved_clip=f.clipping_count.mean(), accuracy=accum["correct"]/accum["n"],
                          **accum))
    if rule != "greedy":
        for target in ["e02", "e04", "e16", "e64", "unlimited"]:
            total = sum(score(raw_losses(family, r.params, groups[sid])[0], FLOORS[target]).sum()
                        for sid, r in f.iterrows())
            transfers.append(dict(family=family, fitted_rule=rule, scoring_rule=target,
                                  mean_nll=total/len(f)))

summary = pd.DataFrame(summaries)
subject = pd.DataFrame(subjects)
transfer = pd.DataFrame(transfers)
summary.to_csv(OUT / "saved_fit_summary.csv", index=False)
subject.to_csv(OUT / "saved_subject_diagnostics.csv", index=False)
transfer.to_csv(OUT / "fixed_parameter_rescoring.csv", index=False)

# Compare each saved optimum against every available fit in the SAME family,
# scored under the SAME objective. This is a direct certificate of missed optima.
candidates, warm = [], []
for family in ["exponential", "hyperbolic", "hyperboloid"]:
    bounds = [(0.01, 20.), (0., 1.)] + ([(0., 5.)] if family == "hyperboloid" else [])
    for sid, g in groups.items():
        available = [(rule, f.loc[sid, "params"]) for (fam, rule), f in fits.items()
                     if fam == family and rule != "greedy"]
        for target in ["e02", "e04", "e16", "e64", "unlimited"]:
            scores = [(score(raw_losses(family, p, g)[0], FLOORS[target]).sum(), rule)
                      for rule, p in available]
            best, source = min(scores)
            saved = fits[family, target].loc[sid, "best_nll"]
            candidates.append(dict(family=family, participant_id=sid, target=target,
                                   saved_nll=saved, best_candidate_nll=best,
                                   source=source, improvement=saved-best))
        # Diagnostic refit only: initialize with good already-saved solutions.
        eps = .01
        objective = lambda p: score(raw_losses(family, p, g)[0], eps).sum()
        scored = sorted([(objective(p), i) for i, (_, p) in enumerate(available)])
        start_indices = {scored[0][1], next(i for i, (rule, _) in enumerate(available) if rule == "unlimited")}
        best_score, best_idx = scored[0]
        best_p = available[best_idx][1].copy()
        for idx in start_indices:
            result = minimize(objective, available[idx][1], method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-7})
            if result.fun < best_score:
                best_score, best_p = result.fun, result.x
        loss, z = raw_losses(family, best_p, g)
        saved = fits[family, "e02"].loc[sid, "best_nll"]
        extreme = np.abs(z) > np.log(99.)
        correct = (z > 0) == g.y.to_numpy()
        warm.append(dict(family=family, participant_id=sid, saved_nll=saved,
                         warm_nll=best_score, improvement=saved-best_score, n=len(g),
                         correct=correct.sum(), clipped=extreme.sum(),
                         extreme_correct=np.sum(extreme & correct), extreme_wrong=np.sum(extreme & ~correct),
                         raw_nll=loss.sum(), parameters=json.dumps(best_p.tolist())))

candidate = pd.DataFrame(candidates)
warm = pd.DataFrame(warm)
candidate.to_csv(OUT / "same_objective_candidate_checks.csv", index=False)
warm.to_csv(OUT / "warm_start_e02_diagnostic.csv", index=False)
print("Max per-trial replay error from rounded saved parameters:", max_replay_error)
print("\nSAVED SUMMARY\n", summary.to_string(index=False))
print("\nRESCORING\n", transfer.pivot(index=["family", "fitted_rule"], columns="scoring_rule", values="mean_nll").to_string())
print("\nMISSED OPTIMA CERTIFICATES\n", candidate.groupby(["family", "target"]).agg(
    mean_saved=("saved_nll", "mean"), mean_candidate=("best_candidate_nll", "mean"),
    improved_subjects=("improvement", lambda s: (s > .01).sum()),
    mean_gap=("improvement", "mean")).to_string())
print("\nWARM START DIAGNOSTIC\n", warm.groupby("family").agg(
    saved_nll=("saved_nll", "mean"), warm_nll=("warm_nll", "mean"),
    improved=("improvement", lambda s: (s>.01).sum()), correct=("correct", "sum"),
    n=("n", "sum"), clipped=("clipped", "sum"), extreme_correct=("extreme_correct", "sum"),
    extreme_wrong=("extreme_wrong", "sum"), raw_nll=("raw_nll", "mean")).to_string())
print("\nWORST EXPONENTIAL EXAMPLES\n", warm[warm.family == "exponential"].nlargest(5, "improvement").to_string(index=False))
