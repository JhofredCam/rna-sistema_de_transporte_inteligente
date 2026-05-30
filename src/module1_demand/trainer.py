import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_model(
    model,
    train_loader,
    test_loader,
    device,
    epochs=50,
    lr=0.001,
    patience=10,
    clip_norm=1.0,
    output_dir="demand_prediction",
    seed=1234,
):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
        factor=0.5,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0

        for X_batch, routes_batch, climas_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            routes_batch = routes_batch.to(device)
            climas_batch = climas_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch, routes_batch, climas_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, routes_batch, climas_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                routes_batch = routes_batch.to(device)
                climas_batch = climas_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch, routes_batch, climas_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)

        print(
            f"Epoch [{epoch+1:2d}/{epochs}]  "
            f"Train: {avg_train_loss:.4f}  "
            f"Val: {avg_val_loss:.4f}  "
            f"LR: {current_lr:.2e}"
        )

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            os.makedirs(output_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                f"{output_dir}/best_model.pth",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹ Early stopping en epoch {epoch + 1}")
                break

    return history
