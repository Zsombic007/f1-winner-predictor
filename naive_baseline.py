import pandas as pd

DATA_PATH = "data/f1_results.csv"

def main():
    results = pd.read_csv(DATA_PATH)

    naive_accuracy = (results["winner"] == 0).mean()
    pole_picks = results[results["grid_position"] == 1]
    pole_hit_rate = pole_picks["winner"].mean()

    print(f"All-negative predictor")
    print(f"Row accuracy: {naive_accuracy:.1%}")
    print(f"Winners selected: 0")

    print("\nPole-sitter baseline")
    print(f"Race-level hit rate: {pole_hit_rate:.1%}")
    print(f"Race evaluated: {len(pole_picks)}")

    print("\nThe first number looks impressive but never picks a race winner. The race-level hit rate answers the real question")

if __name__ == "__main__":
    main()