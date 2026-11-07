
import pandas as pd

INPUT_PATH = "data/f1_results.csv"
OUTPUT_PATH = "data/f1_features.csv"
WINDOW = 5
FEATURE_COLUMNS = [
    "grid_position",
    "driver_recent_finish",
    "driver_recent_points",
    "constructor_recent_finish",
    "constructor_recent_points"
]

def prior_rolling_mean(series):
    return series.shift(1).rolling(WINDOW, min_periods = 1).mean()

def main():
    results = pd.read_csv(INPUT_PATH, parse_dates = ["date"])
    results = results.sort_values(
        ["date", "race_id", "finish_position"]
    ).reset_index(drop = True)

    driver_groups = results.groupby("driver_id", sort = False)
    results["driver_recent_finish"] = driver_groups["finish_position"].transform(prior_rolling_mean)

    results["driver_recent_points"] = driver_groups["points"].transform(prior_rolling_mean)

    constructor_races=(
        results.groupby(["race_id", "date", "constructor_id"], as_index=False,sort=False)
    .agg(constructor_finish=("finish_position", "mean"), constructor_points=("points", "sum"),
         ).sort_values(["date", "race_id", "constructor_id"]))

    constructor_groups = constructor_races.groupby("constructor_id", sort=False)
    constructor_races["constructor_recent_finish"] = constructor_groups["constructor_finish"].transform(prior_rolling_mean)
    constructor_races["constructor_recent_points"] = constructor_groups["constructor_points"].transform(prior_rolling_mean)

    constructor_features = constructor_races[
        [
            "race_id",
            "date",
            "constructor_id",
            "constructor_recent_finish",
            "constructor_recent_points"
        ]
    ]

    features = results.merge(
        constructor_features,
        how="left",
        on=["race_id", "date", "constructor_id"],
        validate="many_to_one"
    )
    features.to_csv(OUTPUT_PATH, index=False)

    sample=features.dropna(subset=FEATURE_COLUMNS).tail(8)
    print(f"Saved {len(features)} feature rows to {OUTPUT_PATH}.")
    print(sample[["race_name", "driver_name", *FEATURE_COLUMNS]].to_string(index=False))

if __name__ == "__main__":
    main()