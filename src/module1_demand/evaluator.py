import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def denormalize(values, target_scaler):
    pas_min = target_scaler.data_min_[0]
    pas_max = target_scaler.data_max_[0]
    return values * (pas_max - pas_min) + pas_min


def compute_metrics_by_route(
    all_preds,
    all_reals,
    all_routes,
    route_encoder,
    target_scaler=None,
):
    if target_scaler is not None:
        reals_real = denormalize(all_reals, target_scaler)
        preds_real = denormalize(all_preds, target_scaler)
    else:
        reals_real = all_reals
        preds_real = all_preds

    resultados = []

    for route_id in sorted(np.unique(all_routes)):
        mask = all_routes == route_id
        y_r = reals_real[mask]
        y_p = preds_real[mask]

        rmse = np.sqrt(mean_squared_error(y_r, y_p))
        mae = mean_absolute_error(y_r, y_p)
        mape = np.mean(np.abs((y_r - y_p) / (y_r + 1e-8))) * 100

        nombre_ruta = route_encoder.inverse_transform([route_id])[0]

        resultados.append({
            "ruta": nombre_ruta,
            "route_id": route_id,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "y_real": y_r,
            "y_pred": y_p,
        })

    return resultados


def compute_global_metrics(reals_real, preds_real):
    rmse = np.sqrt(mean_squared_error(reals_real, preds_real))
    mae = mean_absolute_error(reals_real, preds_real)
    mape = np.mean(
        np.abs((reals_real - preds_real) / (reals_real + 1e-8))
    ) * 100
    return rmse, mae, mape


def print_metrics_table(resultados, reals_real, preds_real):
    print("\n" + "=" * 60)
    print("      MÉTRICAS DE EVALUACIÓN POR RUTA")
    print("=" * 60)

    print(
        f"{'Ruta':<12}"
        f"{'RMSE':>12}"
        f"{'MAE':>12}"
        f"{'MAPE (%)':>14}"
    )
    print("-" * 60)

    for r in resultados:
        print(
            f"{r['ruta']:<12}"
            f"{r['rmse']:>12.2f}"
            f"{r['mae']:>12.2f}"
            f"{r['mape']:>14.2f}%"
        )

    rmse_g, mae_g, mape_g = compute_global_metrics(reals_real, preds_real)

    print("-" * 60)
    print(
        f"{'GLOBAL':<12}"
        f"{rmse_g:>12.2f}"
        f"{mae_g:>12.2f}"
        f"{mape_g:>14.2f}%"
    )
    print("=" * 60)

    return rmse_g, mae_g, mape_g


def plot_predictions(resultados, save_path=None):
    import matplotlib.pyplot as plt
    num_rutas = len(resultados)

    fig, axes = plt.subplots(
        num_rutas,
        1,
        figsize=(14, 4 * num_rutas),
    )

    if num_rutas == 1:
        axes = [axes]

    for ax, r in zip(axes, resultados):
        n_puntos = len(r["y_real"])
        x = np.arange(n_puntos)

        ax.plot(x, r["y_real"], label="Real", linewidth=1.2, alpha=0.9)
        ax.plot(
            x,
            r["y_pred"],
            label="Predicción",
            linewidth=1.2,
            linestyle="--",
            alpha=0.85,
        )
        ax.fill_between(x, r["y_real"], r["y_pred"], alpha=0.12)

        ax.set_title(
            f"{r['ruta']}  —  "
            f"RMSE: {r['rmse']:.1f} | "
            f"MAE: {r['mae']:.1f} | "
            f"MAPE: {r['mape']:.1f}%",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Muestra futura")
        ax.set_ylabel("Pasajeros")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)

    plt.suptitle(
        "Predicción vs Demanda Real por Ruta",
        fontsize=14,
        fontweight="bold",
        y=1.005,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)

    plt.show()

    if save_path:
        print(f"\n✓ Gráfica guardada: {save_path}")


def plot_metrics_comparison(df_metricas, save_path=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metricas = ["rmse", "mae", "mape"]
    titulos = [
        "RMSE (pasajeros)",
        "MAE (pasajeros)",
        "MAPE (%)",
    ]

    for ax, metrica, titulo in zip(axes, metricas, titulos):
        valores = df_metricas[metrica].values
        rutas = df_metricas["ruta"].values

        bars = ax.bar(rutas, valores)
        media = valores.mean()

        ax.axhline(
            media,
            linestyle="--",
            linewidth=1.2,
            label=f"Media: {media:.1f}",
        )

        for bar, val in zip(bars, valores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + media * 0.01,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title(titulo, fontsize=11, fontweight="bold")
        ax.set_xlabel("Ruta")
        ax.set_ylabel(titulo)
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.4)

    plt.suptitle(
        "Comparativa de Métricas por Ruta",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)

    plt.show()

    if save_path:
        print(f"✓ Gráfica guardada: {save_path}")


def plot_heatmap_error(resultados, save_path=None, n_muestras=100):
    import matplotlib.pyplot as plt
    import seaborn as sns
    n = min(
        n_muestras,
        min(len(r["y_real"]) for r in resultados),
    )

    error_matrix = np.array([
        np.abs(r["y_real"][:n] - r["y_pred"][:n])
        for r in resultados
    ])

    fig, ax = plt.subplots(figsize=(16, 4))

    sns.heatmap(
        error_matrix,
        ax=ax,
        cmap="YlOrRd",
        yticklabels=[r["ruta"] for r in resultados],
        xticklabels=False,
        cbar_kws={
            "label": "Error Absoluto (pasajeros)",
        },
    )

    ax.set_title(
        "Heatmap de Error Absoluto "
        "— Primeras 100 Muestras Futuras",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Muestra futura")
    ax.set_ylabel("Ruta")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)

    plt.show()

    if save_path:
        print(f"✓ Gráfica guardada: {save_path}")


def build_summary_df(resultados):
    df_metricas = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("y_real", "y_pred")}
        for r in resultados
    ])

    print("\nTabla resumen:")
    print(df_metricas.to_string(index=False))

    return df_metricas
