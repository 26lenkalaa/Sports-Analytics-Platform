import argparse
import csv
import math
from datetime import date, datetime
from pathlib import Path
from typing import TypedDict


class Match(TypedDict):
    played_on: date
    home: str
    away: str
    home_goals: int
    away_goals: int


MINIMUM_GAMES: int = 5
HALF_LIFE_DAY: float = 90.0


def parse_date(text: str) -> date:
    return datetime.strptime(text.strip(), "%d/%m/%Y").date()


def load_history(source: Path, match_date: date) -> list[Match]:
    """Read results from the preceding calendar year."""
    try:
        start_date: date = match_date.replace(year=match_date.year - 1)
    except ValueError:
        start_date = match_date.replace(
            year=match_date.year - 1,
            day=28,
        )

    files: list[Path] = [
        source / "Premier_League_25-26.csv",
        source / "Premier_League_26-27.csv",
    ]

    for filename in files:
        if not filename.is_file():
            raise FileNotFoundError(f"File not found: {filename}")

    matches: list[Match] = []
    earliest_date: date = date.max
    seen: set[tuple[date, str, str]] = set()

    for filename in files:
        with filename.open(newline="", encoding="utf-8-sig") as file:
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


def match_weight(played_on: date, match_date: date) -> float:
    """Recent matches receive more weight."""
    days_old: int = (match_date - played_on).days

    return math.exp(
        -math.log(2.0) * days_old / HALF_LIFE_DAY
    )


def team_rates(matches: list[Match], team: str, at_home: bool, league_scored: float, league_conceded: float, match_date: date) -> tuple[float, float]:
    """Return weighted scoring and conceding rates."""
    total_weight: float = 0.0
    scored: float = 0.0
    conceded: float = 0.0

    for match in matches:
        weight: float = match_weight(match["played_on"], match_date)

        if at_home and match["home"] == team:
            total_weight += weight
            scored += match["home_goals"] * weight
            conceded += match["away_goals"] * weight

        elif not at_home and match["away"] == team:
            total_weight += weight
            scored += match["away_goals"] * weight
            conceded += match["home_goals"] * weight

    if total_weight >= MINIMUM_GAMES:
        return scored / total_weight, conceded / total_weight

    # Use league averages when weighted history is limited.
    missing_weight: float = MINIMUM_GAMES - total_weight

    scoring_rate: float = (
        scored + missing_weight * league_scored
    ) / MINIMUM_GAMES

    conceding_rate: float = (
        conceded + missing_weight * league_conceded
    ) / MINIMUM_GAMES

    return scoring_rate, conceding_rate


def expected_goals(matches: list[Match], home_team: str, away_team: str, match_date: date) -> tuple[float, float]:
    """Calculate expected goals using weighted rates."""
    if HALF_LIFE_DAY <= 0:
        raise ValueError("Half-life must be positive.")

    total_weight: float = 0.0
    home_goals: float = 0.0
    away_goals: float = 0.0

    for match in matches:
        weight: float = match_weight(match["played_on"], match_date)

        total_weight += weight
        home_goals += match["home_goals"] * weight
        away_goals += match["away_goals"] * weight

    if total_weight <= 0:
        raise ValueError("No weighted history available.")

    league_home: float = home_goals / total_weight
    league_away: float = away_goals / total_weight

    if league_home <= 0 or league_away <= 0:
        raise ValueError("League scoring averages must be positive.")

    home_scored, home_conceded = team_rates(
        matches, home_team, True, league_home, league_away, match_date
    )

    away_scored, away_conceded = team_rates(
        matches, away_team, False, league_away, league_home, match_date
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


def predict(home_mean: float, away_mean: float) -> tuple[dict[str, float], int, int, float, float]:
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

    best_home_goals: int = 0
    best_away_goals: int = 0
    best_score_probability: float = -1.0

    for home_goals in range(21):
        for away_goals in range(21):
            chance: float = (
                home_chances[home_goals]
                * away_chances[away_goals]
            )

            if chance > best_score_probability:
                best_home_goals = home_goals
                best_away_goals = away_goals
                best_score_probability = chance

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

    return (
        probabilities,
        best_home_goals,
        best_away_goals,
        best_score_probability,
        total,
    )


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
        match_date,
    )

    (
        probabilities,
        best_home_goals,
        best_away_goals,
        best_score_probability,
        total,
    ) = predict(
        home_mean,
        away_mean,
    )

    first_used: date = min(match["played_on"] for match in matches)
    last_used: date = max(match["played_on"] for match in matches)

    print(f"First historical match used: {first_used:%d/%m/%Y}")
    print(f"Last historical match used: {last_used:%d/%m/%Y}")
    print(f"Recency half-life: {HALF_LIFE_DAY:g} days")

    print(f"Match: {args.home_team} vs {args.away_team}")
    print(f"Prediction date: {match_date:%d/%m/%Y}")
    print(f"Earlier matches used: {len(matches)}")
    print(f"Home expected goals: {home_mean:.3f}")
    print(f"Away expected goals: {away_mean:.3f}")

    for outcome, probability in probabilities.items():
        print(f"{outcome}: {probability:.2%}")

    print(
        f"Most likely exact score: "
        f"{best_home_goals}-{best_away_goals} "
        f"(home-away)"
    )

    print(
        f"Exact-score probability: "
        f"{best_score_probability / total:.2%}"
    )


if __name__ == "__main__":
    main()
