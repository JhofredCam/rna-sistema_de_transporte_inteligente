import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TransportDataset(Dataset):
    """
    Dataset para series temporales de transporte.

    Retorna:
        X          -> (seq_length, n_features)
        route_id   -> id de la ruta
        clima_id   -> id del clima objetivo
        y          -> demanda objetivo
    """

    def __init__(self, X, routes, climas, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.routes = torch.tensor(routes, dtype=torch.long)
        self.climas = torch.tensor(climas, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.routes[idx],
            self.climas[idx],
            self.y[idx],
        )


def build_sequences(
    df,
    feature_cols,
    seq_length,
    target_col="pasajeros"
):
    """
    Construye ventanas deslizantes.

    Returns:
        X
        routes
        climas
        y
    """

    X = []
    routes = []
    climas = []
    y = []

    for route_id in df["route_id"].unique():

        rdf = (
            df[df["route_id"] == route_id]
            .sort_values("fecha")
        )

        features = rdf[feature_cols].values
        clima_ids = rdf["clima_id"].values
        targets = rdf[target_col].values

        for i in range(len(rdf) - seq_length):

            X.append(
                features[i : i + seq_length]
            )

            routes.append(route_id)

            climas.append(
                clima_ids[i + seq_length]
            )

            y.append(
                targets[i + seq_length]
            )

    return (
        np.array(X),
        np.array(routes),
        np.array(climas),
        np.array(y),
    )


def seed_worker(worker_id, seed=1234):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def create_dataloader(
    dataset,
    batch_size=32,
    shuffle=False,
    seed=1234
):
    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=lambda wid: seed_worker(
            wid,
            seed
        ),
        generator=g,
    )
