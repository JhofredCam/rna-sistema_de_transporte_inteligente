"""EDA auténtico del Módulo 1: predicción de demanda de transporte (LSTM).

Todos los números se calculan directamente desde ``data/demanda_transporte.csv``;
nada se asume de los parámetros del generador sintético. El script genera:

- Figuras PNG en ``docs/figures/module1_demand/``.
- Resumen numérico completo en ``docs/figures/module1_demand/eda_summary.json``
  (consumido luego para reescribir el informe con hallazgos medidos).

Uso (desde la raíz del repo, Windows/PowerShell)::

    .venv\\Scripts\\python.exe scripts\\eda_module1_demand.py

El análisis es determinista (no hay aleatoriedad, por lo que no aplica seed).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin pantalla, para uso en scripts/CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

# ---------------------------------------------------------------------------
# Constantes y rutas (relativas a la raíz del repo, sin argumentos de CLI)
# ---------------------------------------------------------------------------
RAIZ_REPO = Path(__file__).resolve().parents[1]
RUTA_CSV = RAIZ_REPO / "data" / "demanda_transporte.csv"
DIR_SALIDA = RAIZ_REPO / "docs" / "figures" / "module1_demand"

NOMBRES_DIA = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves",
               4: "Viernes", 5: "Sábado", 6: "Domingo"}
NOMBRES_MES = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}

VENTANA_MEDIA_MOVIL = 30      # días para la media móvil de tendencia
N_REZAGOS_ACF = 40            # rezagos calculados para ACF/PACF
Z_95 = 1.96                   # cota del intervalo de confianza al 95 %
PERIODO_DESCOMPOSICION = 7    # estacionalidad semanal (serie diaria)

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _json_nativo(obj):
    """Convierte tipos numpy/pandas a tipos nativos serializables en JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    raise TypeError(f"Tipo no serializable: {type(obj)!r}")


def guardar_figura(fig, nombre: str) -> Path:
    """Guarda una figura como PNG en el directorio de salida."""
    ruta = DIR_SALIDA / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figura] {ruta.relative_to(RAIZ_REPO)} ({ruta.stat().st_size / 1024:.1f} KB)")
    return ruta


# ---------------------------------------------------------------------------
# Carga y validación básica
# ---------------------------------------------------------------------------
def cargar_datos() -> pd.DataFrame:
    """Carga el CSV, ordena por ruta/fecha y deriva la racha de lluvia."""
    df = pd.read_csv(RUTA_CSV, parse_dates=["fecha"])
    df = df.sort_values(["ruta", "fecha"]).reset_index(drop=True)

    # Validación: una fila por día y ruta (requerido para ACF/descomposición)
    for ruta, g in df.groupby("ruta"):
        if not g["fecha"].is_unique:
            raise ValueError(f"Fechas duplicadas en {ruta}; se esperaba serie diaria regular.")

    # Racha de días de lluvia consecutivos (derivada de los datos, por ruta):
    # cumsum sobre los días SIN lluvia crea bloques; el cumsum dentro de cada
    # bloque cuenta 1, 2, 3... días seguidos de lluvia.
    es_lluvia = df["clima"].eq("Lluvia")
    bloque = (~es_lluvia).groupby(df["ruta"]).cumsum()
    df["dias_lluvia_consecutivos"] = es_lluvia.groupby([df["ruta"], bloque]).cumsum()
    return df


# ---------------------------------------------------------------------------
# 1. Estadísticas descriptivas
# ---------------------------------------------------------------------------
def estadisticas_descriptivas(df: pd.DataFrame) -> dict:
    """Conteos, media, std, min, max, mediana y CV de pasajeros (global y por ruta)."""

    def resumen(s: pd.Series) -> dict:
        return {
            "n": int(s.count()),
            "media": float(s.mean()),
            "desviacion_estandar": float(s.std()),
            "min": int(s.min()),
            "q25": float(s.quantile(0.25)),
            "mediana": float(s.median()),
            "q75": float(s.quantile(0.75)),
            "max": int(s.max()),
            "coef_variacion": float(s.std() / s.mean()),
        }

    por_ruta = {ruta: resumen(g["pasajeros"]) for ruta, g in df.groupby("ruta")}
    return {"global": resumen(df["pasajeros"]), "por_ruta": por_ruta}


# ---------------------------------------------------------------------------
# 2 y 3. Estacionalidad semanal y mensual (factores respecto a la media global)
# ---------------------------------------------------------------------------
def estacionalidad(df: pd.DataFrame, columna: str, nombres: dict) -> dict:
    """Promedio de pasajeros por categoría y factor = media_categoría / media_global."""
    media_global = df["pasajeros"].mean()
    global_ = {}
    for valor, g in df.groupby(columna):
        global_[nombres[valor]] = {
            "n": int(len(g)),
            "media": float(g["pasajeros"].mean()),
            "factor": float(g["pasajeros"].mean() / media_global),
        }
    por_ruta = {}
    for ruta, gr in df.groupby("ruta"):
        media_ruta = gr["pasajeros"].mean()
        por_ruta[ruta] = {
            nombres[v]: float(g["pasajeros"].mean() / media_ruta)
            for v, g in gr.groupby(columna)
        }
    return {"media_global_referencia": float(media_global),
            "global": global_, "por_ruta": por_ruta}


# ---------------------------------------------------------------------------
# 4. Impacto del clima (incluye efecto de lluvia consecutiva derivado)
# ---------------------------------------------------------------------------
def impacto_clima(df: pd.DataFrame) -> dict:
    """Factor de demanda por categoría de clima y por racha de lluvia."""
    media_global = df["pasajeros"].mean()

    por_categoria = {}
    for clima, g in df.groupby("clima"):
        por_categoria[clima] = {
            "n": int(len(g)),
            "media": float(g["pasajeros"].mean()),
            "factor": float(g["pasajeros"].mean() / media_global),
        }

    # Efecto de la lluvia consecutiva (racha derivada en cargar_datos)
    def categoria_racha(k: int) -> str:
        if k == 0:
            return "sin_lluvia"
        if k == 1:
            return "lluvia_dia_1"
        if k == 2:
            return "lluvia_dia_2"
        return "lluvia_dia_3_o_mas"

    df = df.assign(racha=df["dias_lluvia_consecutivos"].map(categoria_racha))
    lluvia_consecutiva = {}
    for cat in ["sin_lluvia", "lluvia_dia_1", "lluvia_dia_2", "lluvia_dia_3_o_mas"]:
        g = df.loc[df["racha"] == cat, "pasajeros"]
        lluvia_consecutiva[cat] = {
            "n": int(len(g)),
            "media": float(g.mean()),
            "factor": float(g.mean() / media_global),
        }
    # Efecto incremental medido: día 3+ respecto al primer día de lluvia
    m1 = lluvia_consecutiva["lluvia_dia_1"]["media"]
    m3 = lluvia_consecutiva["lluvia_dia_3_o_mas"]["media"]
    lluvia_consecutiva["efecto_incremental_dia3_vs_dia1_pct"] = float((m3 - m1) / m1 * 100)

    return {"media_global_referencia": float(media_global),
            "por_categoria": por_categoria,
            "lluvia_consecutiva": lluvia_consecutiva}


def efecto_festivo(df: pd.DataFrame) -> dict:
    """Demanda media en festivos vs no festivos (medida desde los datos)."""
    media_global = df["pasajeros"].mean()
    res = {}
    for etiqueta, g in df.groupby("festivo"):
        res["festivo" if etiqueta == 1 else "no_festivo"] = {
            "n": int(len(g)),
            "media": float(g["pasajeros"].mean()),
            "factor": float(g["pasajeros"].mean() / media_global),
        }
    return res


# ---------------------------------------------------------------------------
# 5. Tendencia: media móvil de 30 días + regresión lineal por ruta
# ---------------------------------------------------------------------------
def tendencia(df: pd.DataFrame) -> dict:
    """Pendiente lineal (pasajeros/día y %/año) y R² por ruta; también global."""

    def ajuste(y: np.ndarray) -> dict:
        x = np.arange(len(y), dtype=float)
        pendiente, intercepto = np.polyfit(x, y, 1)
        r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
        media = float(y.mean())
        return {
            "pendiente_pasajeros_por_dia": float(pendiente),
            "pendiente_pasajeros_por_anio": float(pendiente * 365),
            "pendiente_relativa_pct_por_anio": float(pendiente * 365 / media * 100),
            "intercepto": float(intercepto),
            "r2": r2,
        }

    por_ruta = {}
    for ruta, g in df.groupby("ruta"):
        por_ruta[ruta] = ajuste(g["pasajeros"].to_numpy(dtype=float))

    # Tendencia global: promedio de las 5 rutas por fecha
    serie_global = df.groupby("fecha")["pasajeros"].mean()
    return {"ventana_media_movil_dias": VENTANA_MEDIA_MOVIL,
            "por_ruta": por_ruta,
            "global": ajuste(serie_global.to_numpy(dtype=float))}


# ---------------------------------------------------------------------------
# 6. ACF y PACF (ruta representativa)
# ---------------------------------------------------------------------------
def elegir_ruta_representativa(df: pd.DataFrame) -> str:
    """Ruta cuya media de pasajeros es la más cercana a la media global."""
    medias = df.groupby("ruta")["pasajeros"].mean()
    return str((medias - medias.mean()).abs().idxmin())


def acf_pacf(serie: pd.Series, ruta: str) -> dict:
    """ACF/PACF con 40 rezagos; lista los rezagos que superan ±1.96/sqrt(N)."""
    y = serie.to_numpy(dtype=float)
    n = len(y)
    cota = Z_95 / np.sqrt(n)

    valores_acf = acf(y, nlags=N_REZAGOS_ACF, fft=True)
    valores_pacf = pacf(y, nlags=N_REZAGOS_ACF, method="ywm")

    def significativos(valores: np.ndarray) -> list:
        return [{"rezago": int(k), "valor": float(v)}
                for k, v in enumerate(valores) if k > 0 and abs(v) > cota]

    return {
        "ruta": ruta,
        "criterio_eleccion": "media de pasajeros mas cercana a la media global",
        "n_observaciones": int(n),
        "n_rezagos": N_REZAGOS_ACF,
        "cota_significancia_95pct": float(cota),
        "rezagos_significativos_acf": significativos(valores_acf),
        "rezagos_significativos_pacf": significativos(valores_pacf),
        "_valores_acf": valores_acf,   # solo para la figura; se elimina del JSON
        "_valores_pacf": valores_pacf,
    }


# ---------------------------------------------------------------------------
# 7. Descomposición estacional (periodo 7) de la ruta representativa
# ---------------------------------------------------------------------------
def descomposicion(serie: pd.Series, ruta: str):
    """seasonal_decompose aditivo con periodo 7 + fuerzas de tendencia/estacionalidad."""
    dec = seasonal_decompose(serie.to_numpy(dtype=float), model="additive",
                             period=PERIODO_DESCOMPOSICION)
    resid = pd.Series(dec.resid).dropna()
    estacional = pd.Series(dec.seasonal)[resid.index]
    tend = pd.Series(dec.trend)[resid.index]
    observada = pd.Series(dec.observed)[resid.index]

    # Fuerzas al estilo Wang-Smith-Hyndman (1 = componente domina, 0 = puro ruido)
    var_r = resid.var()
    fuerza_estac = float(max(0.0, 1 - var_r / (estacional + resid).var()))
    fuerza_tend = float(max(0.0, 1 - var_r / (tend + resid).var()))

    resumen = {
        "ruta": ruta,
        "modelo": "aditivo",
        "periodo": PERIODO_DESCOMPOSICION,
        "fuerza_tendencia": fuerza_tend,
        "fuerza_estacionalidad_semanal": fuerza_estac,
        "varianza_residuo_pct_sobre_serie": float(var_r / observada.var() * 100),
        "amplitud_estacional_media_abs": float(np.abs(estacional).mean()),
    }
    return resumen, dec


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def fig_serie_tiempo(df: pd.DataFrame):
    rutas = sorted(df["ruta"].unique())
    fig, axes = plt.subplots(len(rutas), 1, figsize=(12, 10), sharex=True)
    for ax, ruta in zip(axes, rutas):
        g = df[df["ruta"] == ruta]
        ax.plot(g["fecha"], g["pasajeros"], lw=0.6, color=sns.color_palette("tab10")[rutas.index(ruta)])
        ax.set_ylabel(ruta, rotation=0, ha="right", va="center")
        ax.set_ylim(bottom=0)
    axes[0].set_title("Serie diaria de pasajeros por ruta (datos crudos)")
    axes[-1].set_xlabel("Fecha")
    fig.tight_layout()
    return guardar_figura(fig, "serie_tiempo_por_ruta.png")


def fig_distribucion(df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ruta, g in df.groupby("ruta"):
        ax1.hist(g["pasajeros"], bins=60, alpha=0.4, density=True, label=ruta)
    ax1.set_title("Histograma de pasajeros por ruta (densidad)")
    ax1.set_xlabel("Pasajeros diarios")
    ax1.set_ylabel("Densidad")
    ax1.legend()
    sns.boxplot(data=df, x="ruta", y="pasajeros", hue="ruta",
                palette="tab10", legend=False, ax=ax2)
    ax2.set_title("Boxplot de pasajeros por ruta")
    ax2.set_xlabel("Ruta")
    ax2.set_ylabel("Pasajeros diarios")
    fig.tight_layout()
    return guardar_figura(fig, "distribucion_pasajeros.png")


def _fig_barras_factores(factores: dict, titulo: str, nombre_archivo: str,
                         xlabel: str) -> Path:
    etiquetas = list(factores.keys())
    vals = [factores[e]["factor"] for e in etiquetas]
    fig, ax = plt.subplots(figsize=(9, 5))
    colores = ["#c0392b" if v < 1 else "#1f77b4" for v in vals]
    barras = ax.bar(etiquetas, vals, color=colores, edgecolor="black", alpha=0.85)
    ax.axhline(1.0, color="black", ls="--", lw=1, label="Media global (factor = 1)")
    for b, v in zip(barras, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Factor respecto a la media global")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.legend()
    fig.tight_layout()
    return guardar_figura(fig, nombre_archivo)


def fig_estacionalidad_semanal(res: dict):
    return _fig_barras_factores(
        res["global"],
        "Estacionalidad semanal medida: factor de demanda por día de la semana",
        "estacionalidad_semanal.png", "Día de la semana")


def fig_estacionalidad_mensual(res: dict):
    return _fig_barras_factores(
        res["global"],
        "Estacionalidad mensual medida: factor de demanda por mes",
        "estacionalidad_mensual.png", "Mes")


def fig_impacto_clima(res: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    cat = res["por_categoria"]
    etiquetas = list(cat.keys())
    vals = [cat[e]["factor"] for e in etiquetas]
    ns = [cat[e]["n"] for e in etiquetas]
    barras = ax1.bar(etiquetas, vals, color=sns.color_palette("tab10")[:len(vals)],
                     edgecolor="black", alpha=0.85)
    ax1.axhline(1.0, color="black", ls="--", lw=1)
    for b, v, n in zip(barras, vals, ns):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}\n(n={n})",
                 ha="center", va="bottom", fontsize=9)
    ax1.set_title("Impacto medido del clima en la demanda")
    ax1.set_ylabel("Factor respecto a la media global")
    ax1.set_ylim(0, max(vals) * 1.25)

    ll = res["lluvia_consecutiva"]
    orden = ["sin_lluvia", "lluvia_dia_1", "lluvia_dia_2", "lluvia_dia_3_o_mas"]
    etiquetas2 = ["Sin lluvia", "Lluvia día 1", "Lluvia día 2", "Lluvia día 3+"]
    vals2 = [ll[k]["factor"] for k in orden]
    ns2 = [ll[k]["n"] for k in orden]
    barras2 = ax2.bar(etiquetas2, vals2, color=sns.color_palette("Blues_d", 4),
                      edgecolor="black", alpha=0.9)
    ax2.axhline(1.0, color="black", ls="--", lw=1)
    for b, v, n in zip(barras2, vals2, ns2):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}\n(n={n})",
                 ha="center", va="bottom", fontsize=9)
    ax2.set_title("Efecto medido de la lluvia consecutiva (racha derivada de los datos)")
    ax2.set_ylabel("Factor respecto a la media global")
    ax2.set_ylim(0, max(vals2) * 1.25)
    fig.tight_layout()
    return guardar_figura(fig, "impacto_clima.png")


def fig_tendencia(df: pd.DataFrame, res_tendencia: dict):
    rutas = sorted(df["ruta"].unique())
    fig, axes = plt.subplots(len(rutas), 1, figsize=(12, 11), sharex=True)
    for i, (ax, ruta) in enumerate(zip(axes, rutas)):
        g = df[df["ruta"] == ruta]
        mm = g["pasajeros"].rolling(VENTANA_MEDIA_MOVIL).mean()
        ax.plot(g["fecha"], mm, lw=1.4, color=sns.color_palette("tab10")[i],
                label=f"Media móvil {VENTANA_MEDIA_MOVIL} días")
        ajuste = res_tendencia["por_ruta"][ruta]
        x = np.arange(len(g))
        ax.plot(g["fecha"], ajuste["intercepto"] + ajuste["pendiente_pasajeros_por_dia"] * x,
                ls="--", color="black", lw=1.2,
                label=f"Tendencia lineal ({ajuste['pendiente_pasajeros_por_anio']:+.0f} pasajeros/año)")
        ax.set_ylabel(ruta, rotation=0, ha="right", va="center")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_title("Tendencia medida por ruta: media móvil de 30 días y regresión lineal")
    axes[-1].set_xlabel("Fecha")
    fig.tight_layout()
    return guardar_figura(fig, "tendencia.png")


def fig_acf_pacf(res: dict):
    cota = res["cota_significancia_95pct"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, clave, titulo in ((ax1, "_valores_acf", "ACF"), (ax2, "_valores_pacf", "PACF")):
        valores = res[clave]
        rezagos = np.arange(len(valores))
        ax.stem(rezagos, valores, basefmt=" ")
        ax.axhspan(-cota, cota, color="gray", alpha=0.2,
                   label=f"IC 95 % (±{cota:.3f})")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{titulo} — {res['ruta']} ({res['n_observaciones']} días)")
        ax.set_xlabel("Rezago (días)")
        ax.legend(fontsize=8)
    fig.suptitle("Autocorrelación medida de la serie de pasajeros", y=1.02)
    fig.tight_layout()
    return guardar_figura(fig, "acf_pacf.png")


def fig_descomposicion(dec, ruta: str):
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for ax, componente, titulo in zip(
            axes,
            [dec.observed, dec.trend, dec.seasonal, dec.resid],
            ["Observada", "Tendencia", "Estacionalidad (periodo 7)", "Residuo"]):
        ax.plot(componente, lw=0.7 if titulo != "Estacionalidad (periodo 7)" else 1.0,
                color="#1f77b4")
        ax.set_ylabel(titulo, rotation=0, ha="right", va="center")
    axes[0].set_title(f"Descomposición estacional aditiva — {ruta}")
    axes[-1].set_xlabel("Día desde el inicio de la serie")
    fig.tight_layout()
    return guardar_figura(fig, "descomposicion_estacional.png")


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Leyendo {RUTA_CSV.relative_to(RAIZ_REPO)} ...")
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    df = cargar_datos()
    print(f"  {len(df)} registros, {df['ruta'].nunique()} rutas, "
          f"{df['fecha'].min():%Y-%m-%d} -> {df['fecha'].max():%Y-%m-%d}")

    # --- Cálculos numéricos -------------------------------------------------
    print("Calculando estadísticas descriptivas ...")
    resumen = {
        "dataset": {
            "archivo": str(RUTA_CSV.relative_to(RAIZ_REPO)),
            "n_registros": int(len(df)),
            "n_rutas": int(df["ruta"].nunique()),
            "rutas": sorted(df["ruta"].unique().tolist()),
            "dias_por_ruta": {r: int(n) for r, n in df.groupby("ruta").size().items()},
            "fecha_inicio": df["fecha"].min(),
            "fecha_fin": df["fecha"].max(),
            "columnas": df.columns.tolist(),
            "valores_nulos": {c: int(v) for c, v in df.isna().sum().items()},
        },
    }
    resumen["estadisticas_pasajeros"] = estadisticas_descriptivas(df)

    print("Midiendo estacionalidad semanal y mensual ...")
    resumen["estacionalidad_semanal"] = estacionalidad(df, "dia_semana", NOMBRES_DIA)
    resumen["estacionalidad_mensual"] = estacionalidad(df, "mes", NOMBRES_MES)

    print("Midiendo impacto del clima y efecto festivo ...")
    resumen["impacto_clima"] = impacto_clima(df)
    resumen["efecto_festivo"] = efecto_festivo(df)

    print("Midiendo tendencia (media móvil + regresión lineal) ...")
    resumen["tendencia"] = tendencia(df)

    ruta_rep = elegir_ruta_representativa(df)
    serie_rep = (df.loc[df["ruta"] == ruta_rep]
                 .sort_values("fecha")["pasajeros"]
                 .reset_index(drop=True))
    print(f"Calculando ACF/PACF y descomposición (ruta representativa: {ruta_rep}) ...")
    res_acf = acf_pacf(serie_rep, ruta_rep)
    resumen["acf_pacf"] = {k: v for k, v in res_acf.items() if not k.startswith("_")}
    res_dec, dec = descomposicion(serie_rep, ruta_rep)
    resumen["descomposicion_estacional"] = res_dec

    # --- Figuras ------------------------------------------------------------
    print("Generando figuras ...")
    figuras = [
        fig_serie_tiempo(df),
        fig_distribucion(df),
        fig_estacionalidad_semanal(resumen["estacionalidad_semanal"]),
        fig_estacionalidad_mensual(resumen["estacionalidad_mensual"]),
        fig_impacto_clima(resumen["impacto_clima"]),
        fig_tendencia(df, resumen["tendencia"]),
        fig_acf_pacf(res_acf),
        fig_descomposicion(dec, ruta_rep),
    ]
    resumen["figuras"] = [str(p.relative_to(RAIZ_REPO)) for p in figuras]

    # --- JSON resumen --------------------------------------------------------
    ruta_json = DIR_SALIDA / "eda_summary.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2, default=_json_nativo)
    print(f"  [json] {ruta_json.relative_to(RAIZ_REPO)} ({ruta_json.stat().st_size / 1024:.1f} KB)")
    print("EDA completado.")


if __name__ == "__main__":
    main()
