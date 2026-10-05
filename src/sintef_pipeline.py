"""Shared, target-safe data and feature pipeline used by experiment notebooks.

The module contains no model training.  It exists so CatBoost and XGBoost
experiments use exactly the same inputs, folds, metric, and submission layout.
"""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score


HORIZON = 168
RANDOM_SEED = 42


@dataclass
class ProjectData:
    root: Path
    X_all: pd.DataFrame
    target_cube: np.ndarray
    units: list[str]
    train_cases: pd.DataFrame
    test_cases: pd.DataFrame
    sample_submission: pd.DataFrame
    feature_sets: dict[str, list[str]]
    feature_group: dict[str, str]
    generator_features: pd.DataFrame
    folds: list[dict]

    @property
    def n_train_cases(self):
        return len(self.train_cases)

    @property
    def n_test_cases(self):
        return len(self.test_cases)

    @property
    def n_all_cases(self):
        return self.n_train_cases + self.n_test_cases


def read_time_csv(path, time_column="date"):
    frame = pd.read_csv(path)
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True)
    return frame


def rows_for_cases(case_ids):
    case_ids = np.asarray(case_ids)
    return (
        case_ids[:, None] * HORIZON
        + np.arange(HORIZON)[None, :]
    ).ravel()


def micro_auc(y_true_cube, y_pred_cube):
    if y_true_cube.shape != y_pred_cube.shape:
        raise ValueError("Truth and prediction cubes must have identical shapes")
    return roc_auc_score(
        y_true_cube.reshape(len(y_true_cube), -1),
        y_pred_cube.reshape(len(y_pred_cube), -1),
        average="micro",
    )


def per_unit_auc(y_true_cube, y_pred_cube, units):
    return {
        unit: roc_auc_score(
            y_true_cube[:, unit_index, :].ravel(),
            y_pred_cube[:, unit_index, :].ravel(),
        )
        for unit_index, unit in enumerate(units)
    }


def make_purged_folds(train_cases, validation_years=(2020, 2021, 2022)):
    starts = pd.DatetimeIndex(train_cases["starttime"])
    folds = []
    for year in validation_years:
        validation_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        validation_end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        validation_cases = np.flatnonzero(
            (starts >= validation_start) & (starts < validation_end)
        )
        training_cases = np.flatnonzero(
            starts + pd.Timedelta(hours=HORIZON - 1) < validation_start
        )
        if not len(training_cases) or not len(validation_cases):
            raise ValueError(f"Empty training or validation partition for {year}")
        if not (
            starts[training_cases].max() + pd.Timedelta(hours=HORIZON - 1)
            < starts[validation_cases].min()
        ):
            raise AssertionError(f"Fold {year} is not temporally purged")
        folds.append({
            "year": year,
            "train_cases": training_cases,
            "validation_cases": validation_cases,
        })
    return folds


def make_submission(data, prediction_cube, filename):
    expected = (data.n_test_cases, len(data.units), HORIZON)
    if prediction_cube.shape != expected:
        raise ValueError(f"Expected predictions {expected}, got {prediction_cube.shape}")

    submission_data = {"Run No": data.test_cases["Run No"].to_numpy()}
    for unit_index, unit in enumerate(data.units):
        for horizon in range(HORIZON):
            submission_data[f"result_committed_{unit}_t{horizon}"] = (
                prediction_cube[:, unit_index, horizon]
            )

    submission = pd.DataFrame(submission_data)[data.sample_submission.columns]
    probabilities = submission.iloc[:, 1:].to_numpy()
    if not submission["Run No"].equals(data.sample_submission["Run No"]):
        raise AssertionError("Run No does not match sample_submission.csv")
    if not np.isfinite(probabilities).all():
        raise ValueError("Submission contains missing or non-finite predictions")
    if probabilities.min() < 0 or probabilities.max() > 1:
        raise ValueError("Submission probabilities are outside [0, 1]")

    path = data.root / "submissions" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return path, submission


def build_project_data(root):
    root = Path(root).resolve()
    data_dir = root / "data"

    price = read_time_csv(data_dir / "kernel" / "Historical_day_ahead_price_2015_2025.csv")
    inflow = read_time_csv(data_dir / "kernel" / "Historical_inflow_1958_2025.csv")
    volume = read_time_csv(data_dir / "kernel" / "Historical_volume_2015_2024.csv")
    water_value = read_time_csv(data_dir / "kernel" / "Synthetic_water_value_2015_2024.csv")
    min_flow = read_time_csv(data_dir / "extended" / "Constraint_min_flow.csv")
    min_volume = read_time_csv(data_dir / "extended" / "Constraint_min_volume.csv")
    targets = read_time_csv(data_dir / "kernel" / "Unit_commitment_decisions.csv", "starttime")
    prediction_mapping = read_time_csv(root / "prediction_mapping.csv", "starttime")
    sample_submission = pd.read_csv(root / "sample_submission.csv")

    with open(data_dir / "extended" / "Tokke_Vinje_topology.yaml") as stream:
        topology = yaml.safe_load(stream)

    pattern = re.compile(r"^result_committed_(.+)_t(\d+)$")
    target_columns = [column for column in targets if pattern.match(column)]
    units = list(dict.fromkeys(pattern.match(column).group(1) for column in target_columns))
    if len(units) != 14 or len(target_columns) != 14 * HORIZON:
        raise AssertionError("Expected 14 units and 168 targets per unit")
    if target_columns != list(sample_submission.columns[1:]):
        raise AssertionError("Target order differs from sample submission")

    target_cube = np.stack([
        targets[[f"result_committed_{unit}_t{hour}" for hour in range(HORIZON)]]
        .to_numpy(dtype=np.int8)
        for unit in units
    ], axis=1)
    if not set(np.unique(target_cube)).issubset({0, 1}):
        raise AssertionError("Targets must be binary")

    train_cases = targets[["Run No", "starttime"]].copy()
    test_cases = prediction_mapping[["Run No", "starttime"]].copy()
    all_cases = pd.concat([train_cases, test_cases], ignore_index=True)
    n_all_cases = len(all_cases)
    starts = pd.DatetimeIndex(all_cases["starttime"])
    horizon_hour = np.tile(np.arange(HORIZON), n_all_cases)
    target_time = pd.DatetimeIndex(
        starts.repeat(HORIZON) + pd.to_timedelta(horizon_hour, unit="h")
    )

    def numeric_indexed(frame):
        return frame.set_index("date").select_dtypes(include=np.number)

    def hourly_matrix(series):
        values = series.reindex(target_time).to_numpy()
        matrix = values.reshape(n_all_cases, HORIZON)
        if not np.isfinite(matrix).all():
            raise ValueError(
                f"Missing/non-finite hourly inputs: {(~np.isfinite(matrix)).sum()}"
            )
        return matrix

    def repeat_case(values):
        values = np.asarray(values)
        if len(values) != n_all_cases:
            raise ValueError("Case-level array has incorrect length")
        return np.repeat(values, HORIZON)

    def cyclic(values, period):
        angle = 2 * np.pi * np.asarray(values) / period
        return np.sin(angle), np.cos(angle)

    def clipped_shift(matrix, offset):
        indices = np.clip(np.arange(HORIZON) + offset, 0, HORIZON - 1)
        return matrix[:, indices]

    def centred_rolling_mean(matrix, window):
        return (
            pd.DataFrame(matrix).T
            .rolling(window=window, min_periods=1, center=True)
            .mean().T.to_numpy()
        )

    feature_data = {}
    feature_group = {}

    def add_feature(name, values, group):
        values = np.asarray(values)
        expected = n_all_cases * HORIZON
        if values.ndim != 1 or len(values) != expected:
            raise ValueError(f"{name}: expected {expected}, got {values.shape}")
        feature_data[name] = values.astype(np.float32)
        feature_group[name] = group

    hour = target_time.hour.to_numpy()
    weekday = target_time.dayofweek.to_numpy()
    month = target_time.month.to_numpy()
    for name, values, period in (
        ("hour", hour, 24),
        ("weekday", weekday, 7),
        ("month", month - 1, 12),
    ):
        sine, cosine = cyclic(values, period)
        add_feature(f"{name}_sin", sine, "calendar")
        add_feature(f"{name}_cos", cosine, "calendar")
    add_feature("horizon_hour", horizon_hour, "calendar")

    price_matrix = hourly_matrix(
        price.set_index("date")["Day-ahead price (EUR/MWh)"]
    )
    price_mean = price_matrix.mean(axis=1)
    price_std = price_matrix.std(axis=1)
    price_rank = pd.DataFrame(price_matrix).rank(axis=1, pct=True).to_numpy()
    add_feature("price", price_matrix.ravel(), "price")
    add_feature("price_week_mean", repeat_case(price_mean), "price")
    add_feature("price_week_std", repeat_case(price_std), "price")
    add_feature("price_week_percentile", price_rank.ravel(), "price")
    add_feature("price_minus_week_mean", (price_matrix - price_mean[:, None]).ravel(), "price")
    add_feature(
        "price_week_zscore",
        ((price_matrix - price_mean[:, None]) / (price_std[:, None] + 1e-6)).ravel(),
        "price_windows",
    )
    add_feature("price_top_10pct", (price_rank >= 0.90).ravel(), "price_windows")
    add_feature("price_bottom_10pct", (price_rank <= 0.10).ravel(), "price_windows")
    for offset in (1, 3, 6, 24):
        lag = clipped_shift(price_matrix, -offset)
        lead = clipped_shift(price_matrix, offset)
        add_feature(f"price_lag_{offset}h", lag.ravel(), "price_windows")
        add_feature(f"price_lead_{offset}h", lead.ravel(), "price_windows")
        add_feature(f"price_change_from_{offset}h", (price_matrix - lag).ravel(), "price_windows")
        add_feature(f"price_change_to_next_{offset}h", (lead - price_matrix).ravel(), "price_windows")
    for window in (3, 6, 12, 24, 48):
        rolling = centred_rolling_mean(price_matrix, window)
        add_feature(f"price_mean_{window}h", rolling.ravel(), "price_windows")
        add_feature(f"price_minus_mean_{window}h", (price_matrix - rolling).ravel(), "price_windows")

    reservoirs = [column for column in volume.columns if column != "date"]
    capacity = {
        name: values["max_vol"]
        for name, values in topology["model"]["reservoir"].items()
    }
    start_volume = numeric_indexed(volume).reindex(starts)[reservoirs]
    start_water = numeric_indexed(water_value).reindex(starts)[reservoirs]
    end_water = numeric_indexed(water_value).reindex(
        starts + pd.Timedelta(days=7)
    )[reservoirs]
    if start_volume.isna().any().any() or start_water.isna().any().any() or end_water.isna().any().any():
        raise ValueError("Missing case-level reservoir data")

    for reservoir in reservoirs:
        fill = start_volume[reservoir].to_numpy() / capacity[reservoir]
        water_start = start_water[reservoir].to_numpy()
        water_end = end_water[reservoir].to_numpy()
        add_feature(f"fill__{reservoir}", repeat_case(fill), "hydrology")
        add_feature(f"water_value_end__{reservoir}", repeat_case(water_end), "water_margins")
        add_feature(f"water_value_change__{reservoir}", repeat_case(water_end - water_start), "water_margins")
        add_feature(f"price_margin__{reservoir}", (price_matrix - water_end[:, None]).ravel(), "water_margins")

    system_fill = (
        start_volume.sum(axis=1).to_numpy()
        / sum(capacity[reservoir] for reservoir in reservoirs)
    )
    water_end_mean = end_water.mean(axis=1).to_numpy()
    add_feature("system_filling_ratio", repeat_case(system_fill), "hydrology")
    add_feature("water_value_end_mean", repeat_case(water_end_mean), "water_margins")
    margin = price_matrix - water_end_mean[:, None]
    add_feature("price_minus_water_value_mean", margin.ravel(), "water_margins")
    add_feature(
        "price_to_abs_water_value_ratio",
        (price_matrix / (np.abs(water_end_mean[:, None]) + 1.0)).ravel(),
        "water_margins",
    )

    inflow_values = numeric_indexed(inflow)
    inflow_matrices = {}
    for column in inflow_values:
        matrix = hourly_matrix(inflow_values[column])
        inflow_matrices[column] = matrix
        add_feature(f"inflow__{column}", matrix.ravel(), "hydrology")
        add_feature(f"inflow_week_mean__{column}", repeat_case(matrix.mean(axis=1)), "hydrology")
    total_inflow = np.sum(list(inflow_matrices.values()), axis=0)
    add_feature("inflow_total", total_inflow.ravel(), "hydrology")
    add_feature("inflow_total_week_mean", repeat_case(total_inflow.mean(axis=1)), "hydrology")
    add_feature("inflow_total_week_std", repeat_case(total_inflow.std(axis=1)), "hydrology")
    add_feature("inflow_total_24h_mean", centred_rolling_mean(total_inflow, 24).ravel(), "hydrology")
    add_feature(
        "inflow_total_cumulative_mean",
        (np.cumsum(total_inflow, axis=1) / (np.arange(HORIZON)[None, :] + 1)).ravel(),
        "hydrology",
    )

    flow_values = numeric_indexed(min_flow)
    min_volume_values = numeric_indexed(min_volume)
    flow_matrices = {}
    min_volume_matrices = {}
    for column in flow_values:
        matrix = hourly_matrix(flow_values[column])
        flow_matrices[column] = matrix
        add_feature(f"min_flow__{column}", matrix.ravel(), "constraints")
    for column in min_volume_values:
        matrix = hourly_matrix(min_volume_values[column])
        min_volume_matrices[column] = matrix
        add_feature(f"min_volume__{column}", matrix.ravel(), "constraints")
        if column in reservoirs:
            initial = start_volume[column].to_numpy()[:, None]
            add_feature(
                f"min_volume_slack__{column}",
                ((initial - matrix) / capacity[column]).ravel(),
                "constraints",
            )

    flow_active_count = np.sum([matrix > 0 for matrix in flow_matrices.values()], axis=0)
    volume_active_count = np.sum([matrix > 0 for matrix in min_volume_matrices.values()], axis=0)
    total_min_flow = np.sum(list(flow_matrices.values()), axis=0)
    total_min_volume = np.sum(list(min_volume_matrices.values()), axis=0)
    add_feature("min_flow_active_count", flow_active_count.ravel(), "constraints")
    add_feature("min_volume_active_count", volume_active_count.ravel(), "constraints")
    add_feature("min_flow_total", total_min_flow.ravel(), "constraints")
    add_feature("min_volume_total", total_min_volume.ravel(), "constraints")

    variable_flow_columns = [
        column for column in flow_values
        if 0 < (flow_values[column] > 0).mean() < 1
    ]
    variable_volume_columns = [
        column for column in min_volume_values
        if 0 < (min_volume_values[column] > 0).mean() < 1
    ]
    variable_active = np.zeros((n_all_cases, HORIZON), dtype=bool)
    for column in variable_flow_columns:
        variable_active |= flow_matrices[column] > 0
    for column in variable_volume_columns:
        variable_active |= min_volume_matrices[column] > 0
    add_feature("variable_constraint_active", variable_active.ravel(), "constraints")

    add_feature("margin_x_system_fill", (margin * system_fill[:, None]).ravel(), "interactions")
    add_feature("price_rank_x_system_fill", (price_rank * system_fill[:, None]).ravel(), "interactions")
    add_feature("margin_x_total_inflow", (margin * total_inflow).ravel(), "interactions")
    add_feature("price_rank_x_hour_sin", price_rank.ravel() * feature_data["hour_sin"], "interactions")
    add_feature("price_rank_x_hour_cos", price_rank.ravel() * feature_data["hour_cos"], "interactions")
    add_feature("margin_x_constraint", margin.ravel() * variable_active.ravel(), "interactions")

    X_all = pd.DataFrame(feature_data)
    constants = [
        column for column in X_all
        if X_all[column].nunique(dropna=False) <= 1
    ]
    X_all = X_all.drop(columns=constants)
    if X_all.isna().any().any() or not np.isfinite(X_all.to_numpy()).all():
        raise ValueError("Final feature matrix contains invalid values")

    baseline = [
        "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
        "month_sin", "month_cos", "horizon_hour", "price",
        "price_week_mean", "price_week_std", "price_week_percentile",
        "price_minus_week_mean", "water_value_end_mean",
        "price_minus_water_value_mean", "system_filling_ratio",
        "inflow_total", "inflow_total_24h_mean", "min_flow_total",
        "min_volume_total", "variable_constraint_active",
    ]

    def grouped(name):
        return [
            feature for feature, group in feature_group.items()
            if group == name and feature in X_all.columns
        ]

    def unique_existing(names):
        return list(dict.fromkeys(name for name in names if name in X_all.columns))

    feature_sets = {
        "baseline": unique_existing(baseline),
        "baseline_plus_price": unique_existing(baseline + grouped("price_windows")),
        "baseline_plus_price_water": unique_existing(
            baseline + grouped("price_windows") + grouped("water_margins")
        ),
        "baseline_plus_price_water_hydrology": unique_existing(
            baseline + grouped("price_windows") + grouped("water_margins")
            + grouped("hydrology")
        ),
        "full_engineered": unique_existing(
            baseline + grouped("price_windows") + grouped("water_margins")
            + grouped("hydrology") + grouped("constraints")
            + grouped("interactions")
        ),
    }

    generator_rows = []
    generator_config = topology["model"]["generator"]
    for unit in units:
        config = generator_config[unit]
        plant = unit.rsplit("_G", 1)[0]
        number_match = re.search(r"_G(\d+)$", unit)
        generator_rows.append({
            "unit": unit,
            "plant": plant,
            "generator_number": int(number_match.group(1)) if number_match else 1,
            "p_min": config.get("p_min", np.nan),
            "p_max": config.get("p_max", np.nan),
            "p_nom": config.get("p_nom", np.nan),
            "minimum_output_fraction": config.get("p_min", np.nan) / config.get("p_max", np.nan),
            "start_cost": config.get("startcost_const", 0.0),
            "stop_cost": config.get("stopcost_const", 0.0),
            "penstock": str(config.get("penstock", "unknown")),
        })
    generator_features = pd.DataFrame(generator_rows)
    generator_features["units_in_plant"] = (
        generator_features.groupby("plant")["unit"].transform("count")
    )

    return ProjectData(
        root=root,
        X_all=X_all,
        target_cube=target_cube,
        units=units,
        train_cases=train_cases,
        test_cases=test_cases,
        sample_submission=sample_submission,
        feature_sets=feature_sets,
        feature_group=feature_group,
        generator_features=generator_features,
        folds=make_purged_folds(train_cases),
    )
