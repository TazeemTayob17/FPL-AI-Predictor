"""Checks the backtest's time-ordered validation split and the fixed-squad captain-doubling arithmetic."""

import pandas as pd

from backtest_season import gameweek_points_with_captain, recency_split

GW_ROWS = pd.DataFrame(
    {
        "season": ["s"] * 10,
        "GW": list(range(1, 11)),
    }
)


def test_recency_split_puts_the_most_recent_rows_in_validation():
    """The last 10% of time-ordered rows (by season, GW) must land in the validation slice, not the training slice."""
    train, valid = recency_split(GW_ROWS, valid_fraction=0.1)
    assert valid["GW"].tolist() == [10]
    assert 10 not in train["GW"].tolist()


def test_recency_split_preserves_row_count():
    """No rows should be gained or dropped by the split."""
    train, valid = recency_split(GW_ROWS, valid_fraction=0.3)
    assert len(train) + len(valid) == len(GW_ROWS)


def test_gameweek_points_doubles_only_the_top_ranked_players_own_points():
    """The captain's own points count twice (once in the squad sum, once as the bonus), no one else's do."""
    gw_rows = pd.DataFrame(
        {"name": ["A", "B", "C"], "total_points": [10, 5, 2], "model_prediction": [8.0, 3.0, 1.0]}
    )
    result = gameweek_points_with_captain(gw_rows, squad_names={"A", "B", "C"}, captaincy_column="model_prediction")
    assert result == 10 + 5 + 2 + 10


def test_gameweek_points_only_counts_squad_members():
    """A high-scoring player outside the fixed squad must not contribute to the total at all."""
    gw_rows = pd.DataFrame(
        {"name": ["A", "B", "Outsider"], "total_points": [10, 5, 50], "model_prediction": [8.0, 3.0, 100.0]}
    )
    result = gameweek_points_with_captain(gw_rows, squad_names={"A", "B"}, captaincy_column="model_prediction")
    assert result == 10 + 5 + 10
