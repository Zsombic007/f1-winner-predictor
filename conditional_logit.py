"""
Conditional logit (McFadden choice model) for F1 race winners.

============================================================================
  AI-GENERATED FILE
  This entire file was written by an AI assistant (Claude, by Anthropic,
  via Claude Code), not by the repository owner. It was added on request
  as an extra model for comparison. Review it like any third-party code.
============================================================================

Why this model?
    The other classifiers in this project score every driver independently
    ("does driver X win?") and only afterwards normalize the scores within a
    race. A conditional logit models the actual question directly: *which one
    driver out of this race's field wins?*

        P(driver i wins race r) = exp(x_i . w) / sum_{j in r} exp(x_j . w)

    The softmax is taken over the drivers of a single race, so the
    probabilities sum to 100% per race by construction, and the model learns
    from comparisons between drivers in the same race, not between drivers in
    different races. There is no intercept: it would cancel out in the softmax.

    The weights are fitted by maximum likelihood with an L2 penalty, using
    L-BFGS and an analytic gradient. Only numpy and scipy are needed.

Usage:
    python conditional_logit.py     # multi-season backtest, same split as compare_models.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from train_predictor import FEATURES

DATA_PATH = "data/f1_features.csv"
TEST_SEASONS = [2023, 2024, 2025]


class ConditionalLogit:
    """Softmax-within-race logit. AI-generated (see the file header)."""

    def __init__(self, l2=1.0):
        self.l2 = l2
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.coef_ = None

    def _design(self, frame, fit):
        values = frame[FEATURES]
        if fit:
            return self.scaler.fit_transform(self.imputer.fit_transform(values))
        return self.scaler.transform(self.imputer.transform(values))

    @staticmethod
    def _race_softmax(utility, race_codes, n_races):
        # Subtract each race's max utility for numerical stability.
        race_max = np.full(n_races, -np.inf)
        np.maximum.at(race_max, race_codes, utility)
        exp = np.exp(utility - race_max[race_codes])
        race_sum = np.bincount(race_codes, weights=exp, minlength=n_races)
        return exp / race_sum[race_codes], race_max, race_sum

    def fit(self, train):
        # Races without exactly one recorded winner carry no usable signal.
        winners_per_race = train.groupby("race_id")["winner"].transform("sum")
        train = train[winners_per_race == 1]

        X = self._design(train, fit=True)
        y = train["winner"].to_numpy(dtype=float)
        race_codes, _ = pd.factorize(train["race_id"])
        n_races = race_codes.max() + 1

        def loss_and_grad(w):
            utility = X @ w
            prob, race_max, race_sum = self._race_softmax(utility, race_codes, n_races)
            log_norm = race_max + np.log(race_sum)  # log-sum-exp per race
            nll = log_norm.sum() - (y * utility).sum()
            grad = X.T @ (prob - y)
            return (nll + 0.5 * self.l2 * w @ w) / n_races, (grad + self.l2 * w) / n_races

        result = minimize(loss_and_grad, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
        if not result.success:
            raise RuntimeError(f"Conditional logit did not converge: {result.message}")
        self.coef_ = result.x
        return self

    def predict(self, test):
        scored = test.copy()
        utility = self._design(test, fit=False) @ self.coef_
        race_codes, _ = pd.factorize(test["race_id"])
        prob, _, _ = self._race_softmax(utility, race_codes, race_codes.max() + 1)
        scored["win_probability"] = prob
        return scored


def conditional_logit_predictor(l2=1.0):
    """Adapter with the same (train, test) -> scored signature as compare_models.py."""
    def run(train, test):
        return ConditionalLogit(l2=l2).fit(train).predict(test)
    return run


def main():
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    total_hits = total_races = 0

    print("Conditional logit (AI-generated model), expanding-window backtest")
    for season in TEST_SEASONS:
        train = data[data["season"] < season]
        test = data[data["season"] == season]
        model = ConditionalLogit().fit(train)
        scored = model.predict(test)

        top_picks = (
            scored.sort_values(["race_id", "win_probability", "grid_position"], ascending=[True, False, True])
            .drop_duplicates("race_id", keep="first")
        )
        hits, n = int(top_picks["winner"].sum()), len(top_picks)
        total_hits += hits
        total_races += n
        winner_prob = scored.loc[scored["winner"] == 1, "win_probability"].mean()
        print(f"  {season}: {hits}/{n} ({hits / n:.1%}), avg prob on winner {winner_prob:.1%}")

    print(f"  Total: {total_hits}/{total_races} ({total_hits / total_races:.1%})")
    print("\nCoefficients from the last fit (standardized features; positive = higher value raises win chance):")
    for name, coef in zip(FEATURES, model.coef_):
        print(f"  {name:28s} {coef:+.3f}")


if __name__ == "__main__":
    main()
