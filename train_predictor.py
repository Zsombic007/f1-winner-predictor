import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/f1_features.csv"
TEST_SEASON = 2025
FEATURES = [
    "grid_position",
    "driver_recent_finish",
    "driver_recent_points",
    "constructor_recent_finish",
    "constructor_recent_points"
]

def normalize_within_race(frame, raw_probabilities):
    scored = frame.copy()
    scored["raw_win_probability"] = raw_probabilities
    race_totals = scored.groupby("race_id")["raw_win_probability"].transform("sum")
    scored["win_probability"] = scored["raw_win_probability"] / race_totals
    return scored

def main():
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    train = data[data["season"] < TEST_SEASON].copy()
    test = data[data["season"] == TEST_SEASON].copy()

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]
    )
    model.fit(train[FEATURES], train["winner"])

    positive_class_index = list(model.classes_).index(1)
    raw_probabilities = model.predict_proba(test[FEATURES])[:,positive_class_index]
    scored = normalize_within_race(test, raw_probabilities)

    ordered = scored.sort_values(["race_id", "win_probability"], ascending=[True, False])
    top_picks = ordered.drop_duplicates("race_id", keep="first")
    pole_picks = scored[scored["grid_position"] == 1]
    actual_winners = scored[scored["winner"] == 1]

    print("2025 chronological backtest")
    print(f"Model top-picks hit rate: {top_picks["winner"].mean():.1%}")
    print(f"Pole-sitter hit rate: {pole_picks["winner"].mean():.1%}")
    print(f"Average probability assigned to the actual winner: {actual_winners["win_probability"].mean():.1%}")

    latest_race_id = scored.sort_values("date").iloc[-1]["race_id"]
    race_table = scored[scored["race_id"] == latest_race_id].sort_values("win_probability", ascending=False)
    display = race_table[
        ["driver_name", "constructor_name", "grid_position", "win_probability", "finish_position"].copy()
    ]
    display["win_probability"] = display["win_probability"].map(lambda value : f"{value:.1%}")

    print(f"\nBacktest race: {race_table.iloc[0]["race_name"]}")
    print(display.to_string(index=False))
    print(f"Probability total: {race_table["win_probability"].sum():.1%}")
    print(f"Actual winner: {race_table.loc[race_table["winner"]==1,"driver_name"].iloc[0]}")
    print("These are comparative model estimates not calibrated betting odds")

if __name__ == "__main__":
    main()
