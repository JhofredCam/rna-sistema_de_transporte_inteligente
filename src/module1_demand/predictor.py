import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def build_future_test_sequences(
    train_df,
    test_df,
    feature_cols,
    seq_length,
):
    X, routes, climas, y = [], [], [], []

    for route_id in train_df["route_id"].unique():

        train_route = train_df[
            train_df["route_id"] == route_id
        ].sort_values("fecha")

        test_route = test_df[
            test_df["route_id"] == route_id
        ].sort_values("fecha")

        if len(test_route) == 0:
            continue

        combined = pd.concat(
            [train_route, test_route],
            ignore_index=True,
        )

        feats = combined[feature_cols].values
        clima_ids = combined["clima_id"].values
        targets = combined["pasajeros"].values

        split_idx = len(train_route)

        for i in range(split_idx, len(combined)):
            start = i - seq_length

            if start < 0:
                continue

            X.append(feats[start:i])
            routes.append(route_id)
            climas.append(clima_ids[i])
            y.append(targets[i])

    return (
        np.array(X),
        np.array(routes),
        np.array(climas),
        np.array(y),
    )


def predict(model, test_loader, device):
    all_preds = []
    all_reals = []
    all_routes = []

    model.eval()

    with torch.no_grad():
        for X_batch, routes_batch, climas_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            routes_batch = routes_batch.to(device)
            climas_batch = climas_batch.to(device)

            outputs = model(X_batch, routes_batch, climas_batch)

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_reals.extend(y_batch.numpy().flatten())
            all_routes.extend(routes_batch.cpu().numpy().flatten())

    return (
        np.array(all_preds),
        np.array(all_reals),
        np.array(all_routes),
    )
