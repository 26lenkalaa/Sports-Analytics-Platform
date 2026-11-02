import argparse
import csv
import math
from datetime import date, datetime
from pathlib import Path
from typing import TypedDict


# These are ordinary dictionaries with defined field types.
class Match(TypedDict):
    played_on: date
    home: str
    away: str
    home_goals: int
    away_goals: int


MINIMUM_GAMES: int = 5


def parse_date(text: str) -> date:
    return datetime.strptime(text, "%d/%m/%Y").date()


def load_history(source: Path, match_date: date) -> list[Match]:
    """Read results from the preceding calendar year."""
    try:
        start_date: date = match_date.replace(year=match_date.year - 1)
    except ValueError:
        # February 29 becomes February 28.
        start_date = match_date.replace(
            year=match_date.year - 1,
            day=28,
        )

    files: list[Path] = [
        source/"Premier League 24-25.csv",
        source/"Premier League 25-26.csv",
    ]

    for filename in files:
        if not filename.is_file():
            raise FileNotFoundError("File" + str(filename) + "not found.")

    matches: list[Match] = []
    earliest_date: date = date.max
    seen: set[tuple[date, str, str]] = set()

    for filename in files:
        with filename.open(
            newline="",
            encoding="utf-8-sig",
        ) as file:
            for row in csv.DictReader(file):
                if not row.get("Date", "").strip():
                    continue

                if row.get("Div", "E0") != "E0":
                    raise ValueError("Use Premier League data only.")

                played_on: date = parse_date(row["Date"])
                earliest_date = min(earliest_date, played_on)

                # Exclude older, same-day and future results.
                if not start_date <= played_on < match_date:
                    continue

                home: str = row["HomeTeam"].strip()
                away: str = row["AwayTeam"].strip()

                key: tuple[date, str, str] = (
                    played_on,
                    home,
                    away,
                )

                if key in seen:
                    raise ValueError(f"Duplicate match: {key}")

                seen.add(key)

                home_goals: int = int(row["FTHG"])
                away_goals: int = int(row["FTAG"])

                if min(home_goals, away_goals) < 0:
                    raise ValueError("Goals cannot be negative.")

                matches.append({
                    "played_on": played_on,
                    "home": home,
                    "away": away,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                })

    if earliest_date > start_date:
        raise ValueError(
            "The files do not extend back a full year. "
            "Include an older season file."
        )

    if not matches:
        raise ValueError("No completed matches in the selected year.")

    return matches


def team_rates(
    matches: list[Match],
    team: str,
    at_home: bool,
    league_scored: float,
    league_conceded: float,
) -> tuple[float, float]:
    """Return scoring and conceding rates for one venue."""
    games: int = 0
    scored: int = 0
    conceded: int = 0

    for match in matches:
        if at_home and match["home"] == team:
            games += 1
            scored += match["home_goals"]
            conceded += match["away_goals"]

        elif not at_home and match["away"] == team:
            games += 1
            scored += match["away_goals"]
            conceded += match["home_goals"]

    if games >= MINIMUM_GAMES:
        return scored / games, conceded / games

    # Use a starting estimate only when history is limited.
    missing_games: int = MINIMUM_GAMES - games

    scoring_rate: float = (
        scored + missing_games * league_scored
    ) / MINIMUM_GAMES

    conceding_rate: float = (
        conceded + missing_games * league_conceded
    ) / MINIMUM_GAMES

    return scoring_rate, conceding_rate


def expected_goals(
    matches: list[Match],
    home_team: str,
    away_team: str,
) -> tuple[float, float]:
    """Calculate expected home and away goals."""
    league_home: float = (
        sum(match["home_goals"] for match in matches)
        / len(matches)
    )

    league_away: float = (
        sum(match["away_goals"] for match in matches)
        / len(matches)
    )

    if league_home <= 0 or league_away <= 0:
        raise ValueError("League scoring averages must be positive.")

    home_scored, home_conceded = team_rates(
        matches, home_team, True, league_home, league_away
    )

    away_scored, away_conceded = team_rates(
        matches, away_team, False, league_away, league_home
    )

    home_attack: float = home_scored / league_home
    away_attack: float = away_scored / league_away

    home_weakness: float = home_conceded / league_away
    away_weakness: float = away_conceded / league_home

    home_mean: float = league_home * home_attack * away_weakness
    away_mean: float = league_away * away_attack * home_weakness

    return home_mean, away_mean


def poisson(goals: int, mean: float) -> float:
    return (
        math.exp(-mean)
        * mean ** goals
        / math.factorial(goals)
    )


def predict(
    home_mean: float,
    away_mean: float,
) -> dict[str, float]:
    """Add scoreline probabilities into three match outcomes."""
    if (
        not math.isfinite(home_mean)
        or not math.isfinite(away_mean)
        or home_mean < 0
        or away_mean < 0
    ):
        raise ValueError("Expected goals must be finite and nonnegative.")
    
    probabilities: dict[str, float] = {
        "Home win": 0.0,
        "Draw": 0.0,
        "Away win": 0.0,
    }

    home_chances: list[float] = [
        poisson(goals, home_mean) for goals in range(21)
    ]

    away_chances: list[float] = [
        poisson(goals, away_mean) for goals in range(21)
    ]

    for home_goals in range(21):
        for away_goals in range(21):
            chance: float = (
                home_chances[home_goals]
                * away_chances[away_goals]
            )

            if home_goals > away_goals:
                probabilities["Home win"] += chance
            elif home_goals == away_goals:
                probabilities["Draw"] += chance
            else:
                probabilities["Away win"] += chance

    total: float = sum(probabilities.values())

    if abs(1 - total) > 1e-8:
        raise ValueError("Increase the goal range in predict().")

    for outcome in probabilities:
        probabilities[outcome] /= total

    return probabilities


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument("source", type=Path)
    parser.add_argument("home_team")
    parser.add_argument("away_team")
    parser.add_argument("match_date")

    args: argparse.Namespace = parser.parse_args()

    if args.home_team == args.away_team:
        raise ValueError("Choose two different teams.")

    match_date: date = parse_date(args.match_date)

    matches: list[Match] = load_history(
        args.source,
        match_date,
    )

    home_mean, away_mean = expected_goals(
        matches,
        args.home_team,
        args.away_team,
    )

    probabilities: dict[str, float] = predict(
        home_mean,
        away_mean,
    )

    first_used: date = min(match["played_on"] for match in matches)
    last_used: date = max(match["played_on"] for match in matches)

    print(f"First historical match used: {first_used:%d/%m/%Y}")
    print(f"Last historical match used: {last_used:%d/%m/%Y}")

    print(f"Match: {args.home_team} vs {args.away_team}")
    print(f"Prediction date: {match_date:%d/%m/%Y}")
    print(f"Earlier matches used: {len(matches)}")
    print(f"Home expected goals: {home_mean:.3f}")
    print(f"Away expected goals: {away_mean:.3f}")

    for outcome, probability in probabilities.items():
        print(f"{outcome}: {probability:.2%}")

    print("Team names must match the CSV exactly.")
    print("Limited-history teams use league-average starting rates.")


if __name__ == "__main__":
    main()
