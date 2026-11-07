from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from compare_models import make_models
from train_predictor import FEATURES, normalize_within_race

DATA_PATH = "data/f1_features.csv"
TEST_SEASON = 2025
PLOT_DIR = Path("plots")

def predict(model, train, test):
    model.fit(train[FEATURES], train["winner"])
    positive = list(model.classes_).index(1)
    raw = model.predict_proba(test[FEATURES])[:, positive]
    return normalize_within_race(test, raw)

def top_picks(scored):
    return(
        scored.sort_values(["race_id", "win_probability"], ascending=[True, False]).drop_duplicates("race_id", keep="first")
    )

def plot_hit_rates(scored_by_model, test):
    rates = {name: top_picks(s)["winner"].mean() for name, s in scored_by_model.items()}
    rates["Pole-sitter"] = test[test["grid_position"] == 1]["winner"].mean()
    rates["Random guess"] = (1 / test.groupby("race_id").size()).mean()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(list(rates.keys()), [v * 100 for v in rates.values()])
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xlabel("Hit rate (%)")
    ax.set_xlim(0, 110)
    ax.invert_yaxis()
    ax.set_title(f"{TEST_SEASON}: how often is the winner predicted?")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "hit_rates.png", dpi=150)
    plt.close(fig)

def plot_winner_probability(scored_by_model):
    n = len(scored_by_model)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(11,4))
    for i, (name, scored) in enumerate(scored_by_model.items()):
        winners = scored[scored["winner"] == 1].sort_values("round")
        offset = (i - (n-1) / 2) * width
        ax.bar(winners["round"]+offset, winners["win_probability"]*100, width=width, label=name)
    ax.set_xlabel("Round")
    ax.set_ylabel("Probablility given to actual winner (%)")
    ax.set_title(f"{TEST_SEASON}: how suprised was the mode?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "winner_probablility.png", dpi=150)
    plt.close(fig)

def top_k_rate(scored, k):
    ranked = scored.sort_values(["race_id", "win_probability"], ascending=[True, False]).copy()
    ranked["rank"] = ranked.groupby("race_id").cumcount() + 1
    winners = ranked[ranked["winner"] == 1]
    return (winners["rank"] <= k).mean()


def plot_top_k(scored_by_model):
    ks = [1, 3, 5]
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, scored in scored_by_model.items():
        ax.plot(ks, [top_k_rate(scored, k) * 100 for k in ks], marker="o", label=name)

    ax.set_xticks(ks)
    ax.set_xlabel("k (top-k picks)")
    ax.set_ylabel("Winner within top-k (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"{TEST_SEASON}: top-k hit rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "top_k.png", dpi=150)
    plt.close(fig)

def main():
    PLOT_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    train = data[data["season"] < TEST_SEASON]
    test = data[data["season"] == TEST_SEASON]
    scored_by_model = {
        name: run(train, test) for name, run in make_models().items()
    }
    plot_hit_rates(scored_by_model, test)
    plot_winner_probability(scored_by_model)
    plot_top_k(scored_by_model)
    print(f"Saved plot to {PLOT_DIR}")

if __name__ == "__main__":
    main()