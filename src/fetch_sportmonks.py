"""Download league results and odds from Sportmonks into res/.

Writes the same CSV columns the model already reads (Date, HomeTeam,
AwayTeam, FTHG, FTAG, AvgCH, AvgCD, AvgCA), so 1yr_predict.py,
5yr_predict.py and backtest.py work unchanged.

Examples:
    python3 fetch_sportmonks.py --check              # which leagues does my token cover?
    python3 fetch_sportmonks.py 2021-08-01 2026-11-10
    python3 fetch_sportmonks.py 2025-08-01 2025-10-31 --league 501 --out res_test
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

BASE_URL = "https://api.sportmonks.com/v3/football"
PREMIER_LEAGUE_ID = 8          # Sportmonks league id; --check lists yours
LEAGUE_NAMES = {8: ("Premier_League", "E0"), 501: ("Scottish_Premiership", "SC0"), 271: ("Superliga", "DK1")}
FULL_TIME_RESULT_MARKET = 1    # the home/draw/away (1X2) odds market
WINDOW_DAYS = 90               # date-range requests are capped, so ask in chunks
FIELDS = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "AvgCH", "AvgCD", "AvgCA"]

JSON = dict[str, Any]


def find_dotenv() -> Path | None:
    """Look for .env next to this script, then in each parent folder (e.g. repo root)."""
    for folder in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        candidate = folder / ".env"
        if candidate.is_file():
            return candidate
    return None

def load_dotenv() -> None:
    """Read KEY=value lines from .env into the environment (if the file exists).

    Real environment variables win, so a token set in the shell or on a
    server is never overwritten by the file.
    """
    path = find_dotenv()
    if path is None:
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

def get_token() -> str:
    load_dotenv()
    token = os.environ.get("SPORTMONKS_TOKEN", "").strip()
    if not token:
        sys.exit(
            "No Sportmonks token found. Copy .env.example to .env and put your token in it."
        )
    return token

def request(path: str, params: dict[str, str], token: str) -> JSON:
    """GET one page. The token goes in a header so it never shows up in URLs or logs."""
    url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Authorization": token, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data: JSON = json.load(response)
            return data
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")[:300]
        if error.code == 429:
            sys.exit("Rate limit hit (free plan: 180 calls/hour per endpoint). Wait and rerun.")
        sys.exit(f"Sportmonks returned {error.code} for {path}: {body}")

def get_all_pages(path: str, params: dict[str, str], token: str) -> list[JSON]:
    """Follow pagination until has_more is false."""
    results: list[JSON] = []
    page = 1
    while True:
        body = request(path, {**params, "page": str(page), "per_page": "50"}, token)
        data = body.get("data") or []
        results.extend(data if isinstance(data, list) else [data])
        if not body.get("pagination", {}).get("has_more"):
            return results
        page += 1
        time.sleep(0.3)  # be polite

def check_access(token: str, league_id: int) -> None:
    """Print the leagues this token can see."""
    leagues = get_all_pages("/leagues", {}, token)
    print(f"Your token covers {len(leagues)} league(s):")
    for league in sorted(leagues, key=lambda item: str(item.get("name"))):
        print(f"  {league.get('id'):>6}  {league.get('name')}")
    if not any(league.get("id") == league_id for league in leagues):
        print(f"\nLeague id {league_id} is NOT in your plan. Pick one from the list with --league.")

def team_names(fixture: JSON) -> tuple[str, str] | None:
    home = away = None
    for team in fixture.get("participants", []):
        location = team.get("meta", {}).get("location")
        if location == "home":
            home = team.get("name")
        elif location == "away":
            away = team.get("name")
    return (home, away) if home and away else None

def final_score(fixture: JSON) -> tuple[int, int] | None:
    """The 'CURRENT' score entries hold the final score once a match is finished."""
    goals: dict[str, int] = {}
    for entry in fixture.get("scores", []):
        if entry.get("description") == "CURRENT":
            score = entry.get("score", {})
            goals[score.get("participant", "")] = int(score.get("goals", 0))
    if "home" in goals and "away" in goals:
        return goals["home"], goals["away"]
    return None

def average_odds(fixture: JSON) -> dict[str, float] | None:
    """Average the 1X2 odds over every bookmaker, like football-data's Avg columns."""
    labels = {"home": "H", "1": "H", "draw": "D", "x": "D", "away": "A", "2": "A"}
    prices: dict[str, list[float]] = {"H": [], "D": [], "A": []}
    for odd in fixture.get("odds", []):
        if odd.get("market_id") != FULL_TIME_RESULT_MARKET:
            continue
        outcome = labels.get(str(odd.get("label", "")).strip().lower())
        try:
            value = float(odd.get("value", ""))
        except ValueError:
            continue
        if outcome and value > 1:
            prices[outcome].append(value)
    if not all(prices.values()):
        return None
    return {o: sum(v) / len(v) for o, v in prices.items()}

def fetch_fixtures(start: date, end: date, token: str, league_id: int) -> list[JSON]:
    fixtures: list[JSON] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=WINDOW_DAYS - 1), end)
        print(f"Fetching {chunk_start} to {chunk_end}...")
        fixtures += get_all_pages(
            f"/fixtures/between/{chunk_start}/{chunk_end}",
            {
                "include": "participants;scores;odds",
                "filters": f"fixtureLeagues:{league_id};markets:{FULL_TIME_RESULT_MARKET}",
            },
            token,
        )
        chunk_start = chunk_end + timedelta(days=1)
    return fixtures

def season_label(day: date) -> str:
    """Matches from August onward belong to the season starting that year (e.g. 26-27)."""
    first = day.year if day.month >= 7 else day.year - 1
    return f"{first % 100:02d}-{(first + 1) % 100:02d}"

def to_rows(fixtures: list[JSON], div: str) -> tuple[list[tuple[date, dict[str, str]]], dict[str, int]]:
    rows: list[tuple[date, dict[str, str]]] = []
    skipped = {"not finished": 0, "no teams": 0, "no odds": 0}
    for fixture in fixtures:
        teams = team_names(fixture)
        score = final_score(fixture)
        if teams is None:
            skipped["no teams"] += 1
            continue
        if score is None or fixture.get("state_id") != 5:  # 5 = full time
            skipped["not finished"] += 1
            continue
        odds = average_odds(fixture)
        if odds is None:
            skipped["no odds"] += 1
        played = datetime.strptime(str(fixture["starting_at"])[:10], "%Y-%m-%d").date()
        rows.append((played, {
            "Div": div,
            "Date": f"{played:%d/%m/%Y}",
            "HomeTeam": teams[0],
            "AwayTeam": teams[1],
            "FTHG": str(score[0]),
            "FTAG": str(score[1]),
            "AvgCH": f"{odds['H']:.2f}" if odds else "",
            "AvgCD": f"{odds['D']:.2f}" if odds else "",
            "AvgCA": f"{odds['A']:.2f}" if odds else "",
        }))
    rows.sort(key=lambda item: item[0])
    return rows, skipped

def write_seasons(rows: list[tuple[date, dict[str, str]]], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_season: dict[str, list[dict[str, str]]] = {}
    for played, row in rows:
        by_season.setdefault(season_label(played), []).append(row)
    for season, season_rows in sorted(by_season.items()):
        path = out_dir / f"{prefix}_{season}.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(season_rows)
        print(f"Wrote {len(season_rows):>3} matches to {path}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("start", nargs="?", help="first date, YYYY-MM-DD")
    parser.add_argument("end", nargs="?", help="last date, YYYY-MM-DD (default: today)")
    parser.add_argument("--out", type=Path, default=Path("res"), help="folder to write CSVs to")
    parser.add_argument("--league", type=int, default=PREMIER_LEAGUE_ID,
                        help="Sportmonks league id (8 = Premier League, 501 = Scottish Premiership)")
    parser.add_argument("--check", action="store_true", help="list the leagues your token covers and stop")
    args = parser.parse_args()

    token = get_token()
    if args.check:
        check_access(token, args.league)
        return
    if not args.start:
        parser.error("give a start date, or use --check")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    prefix, div = LEAGUE_NAMES.get(args.league, (f"League_{args.league}", str(args.league)))
    rows, skipped = to_rows(fetch_fixtures(start, end, token, args.league), div)
    if not rows:
        sys.exit("No finished matches came back. Run --check to confirm your plan covers the league.")
    write_seasons(rows, args.out, prefix)
    print(f"Skipped: {skipped}")
    if skipped["no odds"]:
        print("Some matches have no odds: odds may need an add-on, or may not be kept for old games.")


if __name__ == "__main__":
    main()
