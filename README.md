# 🏎️ F1 Winner Predictor

Can a simple machine-learning model predict the winner of a Formula 1 race **before it starts**, using only information that is known beforehand, and can it beat the obvious guess "the pole-sitter wins"?

This project downloads race results from the Jolpica-F1 API, builds leakage-free features, and compares several models with a time-aware evaluation on the 2023, 2024 and 2025 seasons.

**Short answer:** not convincingly. Over 70 test races the best model (random forest, 60.0%) is one race ahead of the pole-sitter baseline (58.6%), which is well within noise. See [Results](#results).

## Data

- Source: [Jolpica-F1](https://github.com/jolpica/jolpica-f1) (Ergast-compatible API, `https://api.jolpi.ca/ergast/f1`), race results for seasons 2021–2025.
- Licence: the data is licensed under **CC BY-NC-SA 4.0**. It is used here for non-commercial, educational purposes with attribution. If you redistribute data derived from it, keep the same licence.
- Raw API pages are cached in `data/raw/` so the API is never hit twice for the same page. Downloaded and generated data files are **not** committed (see `.gitignore`); you regenerate them with the scripts below.

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install pandas scikit-learn lightgbm matplotlib
```

Python 3.12+ is recommended.

## How to run

Run the scripts in this order:

| Step | Command | What it does |
|---|---|---|
| 1 | `python download_data.py` | Downloads race results (cached) and writes a flat results CSV to `data/`. |
| 2 | `python build_features.py` | Builds the feature table `data/f1_features.csv`. |
| 3 | `python naive_baseline.py` | Prints the "always predict no winner" accuracy and the pole-sitter hit rate. |
| 4 | `python train_predictor.py` | Trains one logistic-regression model, tests on 2025, prints the latest race's win probabilities. |
| 5 | `python compare_models.py` | Compares all models over 2023, 2024 and 2025. |
| 6 | `python plots.py` | Saves the plots in `plots/` (2025 season). |

## Method

**Task.** Each row is one driver in one race, labelled `winner = 1` if they won. Only about 5% of rows are winners, so plain accuracy is meaningless (predicting "nobody wins" is ~95% accurate). The metric used instead is the **top-pick hit rate**: in how many races was the model's highest-ranked driver the actual winner?

**Features** (all known before the race):

- `grid_position`: starting position (pit-lane starters get the last position + 1).
- `driver_recent_finish`, `driver_recent_points`: the driver's rolling mean over their previous 5 races.
- `constructor_recent_finish`, `constructor_recent_points`: the same for the team.

The rolling means use `shift(1)`, so a race never sees its own result (no data leakage).

**Evaluation.** Expanding-window time split: for each test season the models train on all earlier seasons only (2023 trained on 44 races, 2024 on 66, 2025 on 90). There is no random shuffling, which would leak the future into training.

**Models.**

- Logistic regression (median imputation + scaling)
- Gradient boosting
- Random forest
- LightGBM ranker with graded labels (1st/2nd/3rd get 3/2/1 relevance)
- LightGBM ranker with winner-only labels

The per-driver classifiers' probabilities are normalized within each race so they sum to 100%. For the rankers, scores go through a softmax within each race. These are **pseudo-probabilities, not calibrated ones**. Ties are broken by grid position.

**Baselines.** The pole-sitter (the driver starting 1st) and a random guess (the mean of 1/field size, about 5%).

## Results

Top-pick hit rate (winner correctly predicted):

| Model | 2023 (22 races) | 2024 (24 races) | 2025 (24 races) | **Total (70 races)** |
|---|---|---|---|---|
| Pole-sitter (baseline) | 14 (63.6%) | 11 (45.8%) | 16 (66.7%) | **41 (58.6%)** |
| Logistic regression | 19 (86.4%) | 12 (50.0%) | 10 (41.7%) | **41 (58.6%)** |
| Gradient boosting | 15 (68.2%) | 9 (37.5%) | 13 (54.2%) | **37 (52.9%)** |
| Random forest | 18 (81.8%) | 10 (41.7%) | 14 (58.3%) | **42 (60.0%)** |
| LightGBM ranker (graded) | 16 (72.7%) | 12 (50.0%) | 12 (50.0%) | **40 (57.1%)** |
| LightGBM ranker (winners only) | 17 (77.3%) | 9 (37.5%) | 13 (54.2%) | **39 (55.7%)** |

### What the numbers say

- **No model clearly beats the pole-sitter.** The best (random forest) is one race ahead over 70 races. The differences between models are 1–5 races, which is noise at this sample size.
- **The ranking changes every season.** Logistic regression is the best model in 2023 and the worst in 2025. A single season is not enough to pick a winner, which is why the comparison uses three.
- **Season character matters.** 2023 was dominated by one team, which makes the "recent form" features very informative; in 2024 and 2025 the winners were spread across teams and every method gets harder.
- **The top-3 is much easier than the top-1.** In 2025 the actual winner was inside the model's top 3 in 100% of races for most models (91.7% for logistic regression), see `plots/top_k.png`. The hard part is separating the best candidate from the next two.
- **The simplest model is competitive.** Logistic regression ties the pole-sitter overall, so more complex models are not justified by this data.

### Plots (2025 season)

**Hit rate per model vs. baselines**

![Hit rates](plots/hit_rates.png)

**Probability each model gave to the actual winner, per round** (how "surprised" the model was)

![Winner probability](plots/winner_probablility.png)

**Winner within the top-k predictions**

![Top-k](plots/top_k.png)

## Limitations

- **Small sample.** About 110 races in total, 70 in the test set. One race is 1.4 percentage points; model differences are not statistically meaningful.
- **Few features.** Only grid position and rolling form. Missing: track type, weather, tyre strategy, reliability, the car's pace in qualifying vs. rivals.
- **Team rebrands** (e.g. Alfa Romeo → Sauber, AlphaTauri → RB) break the constructor history and produce missing values; rookies have no history. Missing values are median-imputed.
- **Unseeded random forest.** The random-forest numbers change slightly between runs unless `random_state` is fixed.
- **Not calibrated.** The win probabilities are only useful for ranking drivers within a race.
- **Not a betting model.** It does not outperform a trivial baseline, so please do not use it as one.

## Ideas for next steps

- Add qualifying-gap-to-pole, circuit-specific history and weather.
- Fix team aliases so constructor history survives rebrands.
- Repeat random-forest runs over several seeds to measure its stability.
- Compare against bookmaker odds or Elo-style driver/team ratings.

## Project structure

```
download_data.py     # API download with page cache
build_features.py    # leakage-free rolling features
naive_baseline.py    # baselines: all-negative accuracy, pole-sitter
train_predictor.py   # single-model training + shared helpers
ranker.py            # LightGBM learning-to-rank helpers
compare_models.py    # multi-season model comparison
plots.py             # figures in plots/
```

## Acknowledgements

Race data: [Jolpica-F1](https://github.com/jolpica/jolpica-f1), CC BY-NC-SA 4.0. Project based on a NextWork tutorial, extended with caching, baselines, ranking models and time-aware evaluation.
