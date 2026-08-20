# Checks component_models.py's guard rails (too little data / a constant label must degrade to None, not crash) plus one real end-to-end smoke test with enough synthetic data to actually fit.

import numpy as np
import pandas as pd

from fpl_agent.models.component_models import (
    predict_component_models,
    train_all_component_models,
    train_appearance_model,
    train_defensive_contribution_model,
    train_rate_model,
)

COLUMNS = ["difficulty"]


def _synthetic_rows(n: int, position: str, minutes_share: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    minutes = rng.choice([0, 90], size=n, p=[1 - minutes_share, minutes_share])
    goals_scored = np.where(minutes > 0, rng.poisson(0.2, size=n), 0)
    return pd.DataFrame(
        {
            "position": position, "difficulty": rng.uniform(1, 5, size=n), "minutes": minutes,
            "goals_scored": goals_scored, "assists": np.zeros(n, dtype=int), "bonus": np.zeros(n, dtype=int),
        }
    )


# Too few played rows must return None rather than attempting to fit a model on almost nothing.
def test_train_rate_model_returns_none_with_too_few_played_rows():
    train_df = _synthetic_rows(10, "FWD", minutes_share=0.9, seed=1)
    valid_df = _synthetic_rows(3, "FWD", minutes_share=0.9, seed=2)
    assert train_rate_model(train_df, valid_df, COLUMNS, "goals_scored") is None


# A label with only one class (nobody ever played) can't be fit as a binary classifier - must return None, not raise.
def test_train_appearance_model_returns_none_classifiers_when_label_is_constant():
    train_df = _synthetic_rows(30, "MID", minutes_share=0.0, seed=1)
    valid_df = _synthetic_rows(10, "MID", minutes_share=0.0, seed=2)
    result = train_appearance_model(train_df, valid_df, COLUMNS)
    assert result["any_minutes"] is None
    assert result["sixty_plus_minutes"] is None
    assert result["minutes_when_played"] is None


# defensive_contribution only exists for the 2025-26 season in this repo's real data - a features table without that column at all must degrade gracefully, not raise a KeyError.
def test_train_defensive_contribution_model_returns_none_without_the_column():
    train_df = _synthetic_rows(30, "DEF", minutes_share=0.8, seed=1)
    valid_df = _synthetic_rows(10, "DEF", minutes_share=0.8, seed=2)
    assert train_defensive_contribution_model(train_df, valid_df, COLUMNS, threshold=10) is None


# With enough varied real data, the full component set trains without error and predict_component_models runs end to end with sane probability bounds.
def test_train_all_component_models_and_predict_smoke_test():
    train_df = _synthetic_rows(200, "FWD", minutes_share=0.6, seed=1)
    valid_df = _synthetic_rows(40, "FWD", minutes_share=0.6, seed=2)
    models = train_all_component_models(train_df, valid_df, COLUMNS)
    assert "FWD" in models

    features = _synthetic_rows(5, "FWD", minutes_share=0.6, seed=3)
    predicted = predict_component_models(features, models, COLUMNS)
    assert (predicted["p_any_minutes"].between(0, 1)).all()
    assert (predicted["p_sixty_plus_minutes"] <= predicted["p_any_minutes"] + 1e-9).all()


# A position with no trained model at all must still get sane defaults (0 rates, 90-minute fallback) rather than a KeyError.
def test_predict_component_models_defaults_for_an_untrained_position():
    features = pd.DataFrame([{"position": "GKP", "difficulty": 3.0}])
    result = predict_component_models(features, models={}, columns=COLUMNS)
    assert result.loc[0, "p_any_minutes"] == 0.0
    assert result.loc[0, "minutes_when_played"] == 90.0
