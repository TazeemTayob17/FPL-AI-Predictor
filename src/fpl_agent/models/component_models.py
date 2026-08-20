# Trains per-position component models feeding the decomposed points recombination: appearance probabilities (any minutes / 60+ minutes / expected minutes when played), per-90 scoring rates (goals/assists/saves/bonus), and a defensive-contribution threshold probability. Rare events (cards, own goals, penalty misses/saves) deliberately skip a trained model here - they're sparse, mostly-zero events where gradient boosting risks fitting noise, so they're read directly as `{stat}_roll10` features already computed by rolling_stats.py, consistent with how every other per-90-ish signal in this codebase is used.

from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from fpl_agent.models.scoring_rules import DEFENSIVE_CONTRIBUTION_THRESHOLD
from fpl_agent.models.train import LGB_PARAMS, POSITIONS

MIN_TRAIN_ROWS = 20
MIN_VALID_ROWS = 5

CLASSIFIER_PARAMS = {**LGB_PARAMS, "objective": "binary", "metric": "binary_logloss"}
DEFENSIVE_CONTRIBUTION_PARAMS = {**CLASSIFIER_PARAMS, "min_data_in_leaf": 100, "lambda_l1": 1.0, "lambda_l2": 1.0}

# Which per-90 rate models apply to which position - goalkeepers don't get a goals model (real but vanishingly rare, not worth the noise), only goalkeepers get saves.
RATE_STATS = {
    "GKP": {"saves": "saves_per90", "bonus": "bonus_per90"},
    "DEF": {"goals_scored": "goals_per90", "assists": "assists_per90", "bonus": "bonus_per90"},
    "MID": {"goals_scored": "goals_per90", "assists": "assists_per90", "bonus": "bonus_per90"},
    "FWD": {"goals_scored": "goals_per90", "assists": "assists_per90", "bonus": "bonus_per90"},
}


def _train_classifier(X_train, y_train, X_valid, y_valid, params: dict = CLASSIFIER_PARAMS) -> lgb.Booster | None:
    if y_train.nunique() < 2 or y_valid.nunique() < 2:
        return None
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
    return lgb.train(params, train_set, num_boost_round=500, valid_sets=[valid_set], callbacks=[lgb.early_stopping(30, verbose=False)])


def _train_regressor(X_train, y_train, X_valid, y_valid, weight_train=None, weight_valid=None) -> lgb.Booster:
    train_set = lgb.Dataset(X_train, label=y_train, weight=weight_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, weight=weight_valid, reference=train_set)
    return lgb.train(LGB_PARAMS, train_set, num_boost_round=500, valid_sets=[valid_set], callbacks=[lgb.early_stopping(30, verbose=False)])


# One position's appearance sub-model: P(any_minutes), P(60_plus_minutes) classifiers (every row is a valid observation), and E[minutes | played] regression (rows with minutes > 0 only).
def train_appearance_model(train_df: pd.DataFrame, valid_df: pd.DataFrame, columns: list[str]) -> dict:
    any_model = _train_classifier(
        train_df[columns], (train_df["minutes"] > 0).astype(int), valid_df[columns], (valid_df["minutes"] > 0).astype(int)
    )
    sixty_model = _train_classifier(
        train_df[columns], (train_df["minutes"] >= 60).astype(int), valid_df[columns], (valid_df["minutes"] >= 60).astype(int)
    )

    played_train = train_df[train_df["minutes"] > 0]
    played_valid = valid_df[valid_df["minutes"] > 0]
    minutes_model = None
    if len(played_train) >= MIN_TRAIN_ROWS and len(played_valid) >= MIN_VALID_ROWS:
        minutes_model = _train_regressor(played_train[columns], played_train["minutes"], played_valid[columns], played_valid["minutes"])

    return {"any_minutes": any_model, "sixty_plus_minutes": sixty_model, "minutes_when_played": minutes_model}


# One position's per-90 rate model for `stat`: trained only on rows where the player actually played, weighted by minutes/90 exposure so a single cameo appearance doesn't produce a wild, noisy rate with equal say to a 90-minute sample.
def train_rate_model(train_df: pd.DataFrame, valid_df: pd.DataFrame, columns: list[str], stat: str) -> lgb.Booster | None:
    played_train = train_df[(train_df["minutes"] > 0) & train_df[stat].notna()] if stat in train_df.columns else train_df.iloc[0:0]
    played_valid = valid_df[(valid_df["minutes"] > 0) & valid_df[stat].notna()] if stat in valid_df.columns else valid_df.iloc[0:0]
    if len(played_train) < MIN_TRAIN_ROWS or len(played_valid) < MIN_VALID_ROWS:
        return None

    train_nineties = (played_train["minutes"] / 90).clip(lower=0.01)
    valid_nineties = (played_valid["minutes"] / 90).clip(lower=0.01)
    train_rate = played_train[stat] / train_nineties
    valid_rate = played_valid[stat] / valid_nineties
    return _train_regressor(played_train[columns], train_rate, played_valid[columns], valid_rate, train_nineties, valid_nineties)


# One position's defensive-contribution-threshold classifier (DEF: 10+ CBIT, MID/FWD: 12+) - a step function, so modeled as P(threshold met) rather than a smooth rate. Trained only on rows with a real `defensive_contribution` value, which in this repo's data only exists for the 2025-26 season - a genuinely smaller, lower-confidence training set than every other component, flagged here rather than hidden.
def train_defensive_contribution_model(train_df: pd.DataFrame, valid_df: pd.DataFrame, columns: list[str], threshold: int) -> lgb.Booster | None:
    if "defensive_contribution" not in train_df.columns:
        return None
    played_train = train_df[(train_df["minutes"] > 0) & train_df["defensive_contribution"].notna()]
    played_valid = valid_df[(valid_df["minutes"] > 0) & valid_df["defensive_contribution"].notna()]
    if len(played_train) < MIN_TRAIN_ROWS or len(played_valid) < MIN_VALID_ROWS:
        return None

    label_train = (played_train["defensive_contribution"] >= threshold).astype(int)
    label_valid = (played_valid["defensive_contribution"] >= threshold).astype(int)
    return _train_classifier(played_train[columns], label_train, played_valid[columns], label_valid, params=DEFENSIVE_CONTRIBUTION_PARAMS)


# Trains every position's full component set (appearance + per-90 rates + defensive contribution) from one already-time-ordered-split (train_df, valid_df) pair - the same signature shape as train.py's train_position_model, so this slots into both a from-scratch full training run and evaluate.py's per-fold walk-forward retraining.
def train_all_component_models(train_rows: pd.DataFrame, valid_rows: pd.DataFrame, columns: list[str]) -> dict:
    models = {}
    for position in POSITIONS:
        pos_train = train_rows[train_rows["position"] == position]
        pos_valid = valid_rows[valid_rows["position"] == position]
        if len(pos_train) < MIN_TRAIN_ROWS or len(pos_valid) < MIN_VALID_ROWS:
            continue

        appearance = train_appearance_model(pos_train, pos_valid, columns)
        rates = {
            output_column: train_rate_model(pos_train, pos_valid, columns, stat)
            for stat, output_column in RATE_STATS.get(position, {}).items()
        }
        defensive_contribution = None
        if position in DEFENSIVE_CONTRIBUTION_THRESHOLD:
            defensive_contribution = train_defensive_contribution_model(
                pos_train, pos_valid, columns, DEFENSIVE_CONTRIBUTION_THRESHOLD[position]
            )
        models[position] = {"appearance": appearance, "rates": rates, "defensive_contribution": defensive_contribution}
    return models


# Runs every trained component over `features` and attaches its predictions as new columns - the decomposed-model analog of predict.py's predict_with_trained_models.
def predict_component_models(features: pd.DataFrame, models: dict, columns: list[str]) -> pd.DataFrame:
    result = features.copy()
    for column in ("p_any_minutes", "p_sixty_plus_minutes", "goals_per90", "assists_per90", "saves_per90", "bonus_per90", "p_defensive_threshold"):
        result[column] = 0.0
    result["minutes_when_played"] = 90.0  # fallback: assume a full match once actually played, refined below wherever a trained regressor exists

    for position, components in models.items():
        mask = result["position"] == position
        if not mask.any():
            continue
        X = result.loc[mask, columns]

        appearance = components["appearance"]
        if appearance["any_minutes"] is not None:
            result.loc[mask, "p_any_minutes"] = appearance["any_minutes"].predict(X)
        if appearance["sixty_plus_minutes"] is not None:
            result.loc[mask, "p_sixty_plus_minutes"] = appearance["sixty_plus_minutes"].predict(X)
        if appearance["minutes_when_played"] is not None:
            result.loc[mask, "minutes_when_played"] = appearance["minutes_when_played"].predict(X).clip(0, 90)

        for output_column, model in components["rates"].items():
            if model is not None:
                result.loc[mask, output_column] = model.predict(X).clip(0, None)

        if components["defensive_contribution"] is not None:
            result.loc[mask, "p_defensive_threshold"] = components["defensive_contribution"].predict(X)

    result["p_any_minutes"] = result["p_any_minutes"].clip(0, 1)
    result["p_sixty_plus_minutes"] = result["p_sixty_plus_minutes"].clip(0, 1)
    result["p_sixty_plus_minutes"] = result[["p_sixty_plus_minutes", "p_any_minutes"]].min(axis=1)  # 60+ minutes implies any minutes; two independently-trained classifiers can otherwise disagree slightly
    result["p_defensive_threshold"] = result["p_defensive_threshold"].clip(0, 1)
    return result
