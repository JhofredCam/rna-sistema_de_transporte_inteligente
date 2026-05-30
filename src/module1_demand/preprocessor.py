import os

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

FEATURE_COLS_SIN_TARGET = [
    "dia_semana",
    "mes",
    "festivo",
]

TARGET_COL = "pasajeros"


def load_data(csv_path):
    """
    Carga y ordena los datos.
    """

    df = pd.read_csv(csv_path)

    df["fecha"] = pd.to_datetime(df["fecha"])

    df = (
        df
        .sort_values(["ruta", "fecha"])
        .reset_index(drop=True)
    )

    return df


def encode_categorical(df):
    """
    Codifica ruta y clima.
    """

    route_encoder = LabelEncoder()
    clima_encoder = LabelEncoder()

    df["route_id"] = route_encoder.fit_transform(
        df["ruta"]
    )

    df["clima_id"] = clima_encoder.fit_transform(
        df["clima"]
    )

    return (
        df,
        route_encoder,
        clima_encoder,
    )


def temporal_split(
    df,
    pct_train=0.80,
):
    """
    Split temporal por ruta.
    """

    train_parts = []
    test_parts = []

    for ruta in df["ruta"].unique():

        sub = (
            df[df["ruta"] == ruta]
            .sort_values("fecha")
        )

        cutoff = int(
            len(sub) * pct_train
        )

        train_parts.append(
            sub.iloc[:cutoff]
        )

        test_parts.append(
            sub.iloc[cutoff:]
        )

    train_df = pd.concat(
        train_parts
    ).reset_index(drop=True)

    test_df = pd.concat(
        test_parts
    ).reset_index(drop=True)

    return train_df, test_df


def scale_data(
    train_df,
    test_df,
):
    """
    Escala train y test sin leakage.
    """

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[
        FEATURE_COLS_SIN_TARGET
    ] = feature_scaler.fit_transform(
        train_df[
            FEATURE_COLS_SIN_TARGET
        ]
    )

    train_df[
        [TARGET_COL]
    ] = target_scaler.fit_transform(
        train_df[[TARGET_COL]]
    )

    test_df[
        FEATURE_COLS_SIN_TARGET
    ] = feature_scaler.transform(
        test_df[
            FEATURE_COLS_SIN_TARGET
        ]
    )

    test_df[
        [TARGET_COL]
    ] = target_scaler.transform(
        test_df[[TARGET_COL]]
    )

    return (
        train_df,
        test_df,
        feature_scaler,
        target_scaler,
    )


def save_artifacts(
    feature_scaler,
    target_scaler,
    route_encoder,
    clima_encoder,
    output_dir="demand_prediction",
):
    """
    Guarda scalers y encoders.
    """

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    joblib.dump(
        feature_scaler,
        f"{output_dir}/feature_scaler.pkl",
    )

    joblib.dump(
        target_scaler,
        f"{output_dir}/target_scaler.pkl",
    )

    joblib.dump(
        route_encoder,
        f"{output_dir}/route_encoder.pkl",
    )

    joblib.dump(
        clima_encoder,
        f"{output_dir}/clima_encoder.pkl",
    )


def preprocess_pipeline(
    csv_path,
    pct_train=0.80,
    save=True,
    max_year=None,
    output_dir="demand_prediction",
):
    """
    Pipeline completo.
    """

    df = load_data(csv_path)
    if max_year is not None:
        df = df[df["fecha"].dt.year <= max_year]

    (
        df,
        route_encoder,
        clima_encoder,
    ) = encode_categorical(df)

    (
        train_df,
        test_df,
    ) = temporal_split(df,pct_train=pct_train,) # type: ignore

    (
        train_df,
        test_df,
        feature_scaler,
        target_scaler,
    ) = scale_data(
        train_df,
        test_df,
    )

    if save:
        save_artifacts(
            feature_scaler,
            target_scaler,
            route_encoder,
            clima_encoder,
            output_dir=output_dir,
        )

    feature_cols = (
        FEATURE_COLS_SIN_TARGET
        + [TARGET_COL]
    )

    return {
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "route_encoder": route_encoder,
        "clima_encoder": clima_encoder,
    }
