import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_predictor import FEATURES, normalize_within_race

DATA_PATH = "data/f1_features.csv"
TEST_SEASONS = [2023, 2024, 2025]


def make_models():
    return {
        "Logistic regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]),
        "Gradient boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.05,
                max_depth=2, random_state=42,
            )),
        ]),
    }


def evaluate(model, train, test):
    model.fit(train[FEATURES], train["winner"])
    positive = list(model.classes_).index(1)
    raw = model.predict_proba(test[FEATURES])[:, positive]
    scored = normalize_within_race(test, raw)

    top_picks = (
        scored.sort_values(["race_id", "win_probability"], ascending=[True, False])
        .drop_duplicates("race_id", keep="first")
    )
    winners = scored[scored["winner"] == 1]
    return top_picks["winner"].sum(), len(top_picks), winners["win_probability"].mean()


def main():
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    hits = {}
    races_total = 0
    pole_hits = 0

    for season in TEST_SEASONS:
        train = data[data["season"] < season]
        test = data[data["season"] == season]
        n_races = test["race_id"].nunique()
        races_total += n_races

        pole = test[test["grid_position"] == 1]["winner"].sum()
        pole_hits += pole
        print(f"\n{season} ({n_races} races, trained on {train['race_id'].nunique()} races)")
        print(f"  Pole-sitter: {pole}/{n_races}")

        for name, model in make_models().items():
            hit, n, prob = evaluate(model, train, test)
            hits[name] = hits.get(name, 0) + hit
            print(f"  {name}: {hit}/{n} ({hit / n:.1%}), avg prob on winner {prob:.1%}")

    print(f"\nTotal over {races_total} races")
    print(f"  Pole-sitter: {pole_hits / races_total:.1%}")
    for name, hit in hits.items():
        print(f"  {name}: {hit / races_total:.1%}")
    print("Keep the simpler model unless the challenger improves the held-out result.")


if __name__ == "__main__":
    main()