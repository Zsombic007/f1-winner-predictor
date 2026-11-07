import json
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

BASE_URL = "https://api.jolpi.ca/ergast/f1"
SEASONS = range(2021, 2026)
PAGE_SIZE = 100
USER_AGENT = "NextWorkF1Predictor/1.0"
OUTPUT_PATH = Path("data/f1_results.csv")
CACHE_DIR = Path("data/raw")
RETRY_STATUS = {429, 500, 502, 503, 504}


def fetch_page(season, offset, retries=3):
    cache_file = CACHE_DIR / f"{season}_{offset}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    query = urlencode({"limit": PAGE_SIZE, "offset": offset})
    url = f"{BASE_URL}/{season}/results/?{query}"
    request = Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.load(response)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(payload))
            print(f"{season}/{offset}: fetched from API")
            return payload
        except HTTPError as e:
            if e.code not in RETRY_STATUS or attempt == retries:
                raise
            retry_after = e.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else 5 * attempt
            print(f"{season}/{offset}: HTTP {e.code}, waiting {wait}s ({attempt}/{retries})")
            sleep(wait)
        except (URLError, TimeoutError) as e:
            if attempt == retries:
                raise
            print(f"{season}/{offset}: {e}, retrying ({attempt}/{retries})")
            sleep(5 * attempt)


def parse_page(payload):
    rows = []
    races = payload["MRData"]["RaceTable"]["Races"]

    for race in races:
        for result in race["Results"]:
            constructor = result.get("Constructor", {})
            driver = result["Driver"]
            season = int(race["season"])
            round_number = int(race["round"])

            rows.append({
                "race_id": f"{season}-{round_number:02d}",
                "season": season,
                "round": round_number,
                "date": race["date"],
                "race_name": race["raceName"],
                "circuit_id": race["Circuit"]["circuitId"],
                "driver_id": driver["driverId"],
                "driver_name": f"{driver['givenName']} {driver['familyName']}",
                "constructor_id": constructor.get("constructorId", "unknown"),
                "constructor_name": constructor.get("name", "unknown"),
                "grid": int(result.get("grid", "0")),
                "finish_position": int(result["position"]),
                "points": float(result["points"]),
                "winner": int(result["position"] == "1"),
            })

    return rows


def main():
    all_rows = []
    for season in SEASONS:
        offset = 0
        while True:
            cached = (CACHE_DIR / f"{season}_{offset}.json").exists()
            payload = fetch_page(season, offset)
            page_rows = parse_page(payload)
            all_rows.extend(page_rows)

            total = int(payload["MRData"]["total"])
            offset += len(page_rows)
            print(f"{season}: have {offset} of {total} driver results")
            if not page_rows or offset >= total:
                break
            if not cached:
                sleep(0.35)

    results = pd.DataFrame(all_rows)
    results = results.sort_values(
        ["date", "race_id", "finish_position"]
    ).reset_index(drop=True)

    maximum_grid = results.groupby("race_id")["grid"].transform("max")
    results["grid_position"] = results["grid"].where(
        results["grid"] > 0, maximum_grid + 1
    )

    output_columns = [
        "race_id", "season", "round", "date", "race_name", "circuit_id",
        "driver_id", "driver_name", "constructor_id", "constructor_name",
        "grid_position", "finish_position", "points", "winner",
    ]
    results = results[output_columns]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    latest_race_id = results.iloc[-1]["race_id"]
    latest_race = results[results["race_id"] == latest_race_id]
    print(f"\nSaved {len(results)} rows across {results['race_id'].nunique()} races.")
    print(f"Latest cached race: {latest_race.iloc[0]['race_name']}")
    print(
        latest_race[["grid_position", "driver_name", "constructor_name", "finish_position"]]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()