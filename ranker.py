import numpy as np
from lightgbm import LGBMRanker

from train_predictor import FEATURES

def graded_label(frame):
    return (4 - frame["finish_position"]).clip(lower=0)

def fit_ranker(train, params, graded = True):
    train = train.sort_values(["date", "race_id"]).copy()
    labels = graded_label(train) if graded else train["winner"]
    group_sizes = train.groupby("race_id", sort=False).size().to_numpy()

    ranker = LGBMRanker(objective="lambdarank", random_state=42, verbose=-1, **params)
    ranker.fit(train[FEATURES], labels, group=group_sizes)
    return ranker

def predict_ranker(ranker,test):
    scored= test.copy()
    scored["score"] = ranker.predict(test[FEATURES])
    best = scored.groupby("race_id")["score"].transform("max")
    exp = np.exp(scored["score"] - best)
    scored["win_probability"] = exp / exp.groupby(scored["race_id"]).transform("sum")
    return scored