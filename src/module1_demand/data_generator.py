import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

# ============================================================
# CONFIGURACIÓN
# ============================================================

SEED = 1234

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

NUM_DIAS = 1500
FECHA_INICIO = datetime(2024, 1, 1)

RUTAS = ["Ruta A", "Ruta B", "Ruta C", "Ruta D", "Ruta E"]

# ============================================================
# PARÁMETROS DE GENERACIÓN
# ============================================================

# --- Demanda base y tendencia ---
DEMANDA_BASE = {"Ruta A": 1200, "Ruta B": 1800, "Ruta C": 800,
                "Ruta D": 1500, "Ruta E": 1000}
TREND_DRIFT = {"Ruta A": 0.10, "Ruta B": 0.15, "Ruta C": 0.05,
               "Ruta D": 0.12, "Ruta E": 0.08}

# --- Estacionalidad semanal (base + ruido por ocurrencia) ---
# La estructura base define el patrón general, pero cada día concreto
# tiene un multiplicador ligeramente distinto (como en la vida real)
FACTOR_SEMANA_BASE = {0: 1.20, 1: 1.25, 2: 1.25, 3: 1.20, 4: 1.15,
                     5: 0.80, 6: 0.70}
WEEKLY_NOISE_STD = 0.06  # ±6% de variación día a día para el mismo día_semana

# --- Estacionalidad mensual (base + ruido) ---
FACTOR_MES_BASE = {
    1: 0.95, 2: 0.95, 3: 1.00, 4: 1.00, 5: 1.05,
    6: 1.30, 7: 1.40, 8: 1.15,
    9: 1.00, 10: 1.00, 11: 0.95, 12: 1.35
}
MONTHLY_NOISE_STD = 0.04

# --- Clima estacional con persistencia ---
# Probabilidad base por mes (sol/lluvia/nublado)
# Además, si llovió ayer, hoy es más probable que llueva
CLIMA_PROBS = {
     1: (0.60, 0.15, 0.25),  2: (0.55, 0.20, 0.25),
     3: (0.50, 0.25, 0.25),  4: (0.40, 0.35, 0.25),
     5: (0.40, 0.35, 0.25),  6: (0.50, 0.20, 0.30),
     7: (0.60, 0.10, 0.30),  8: (0.60, 0.10, 0.30),
     9: (0.50, 0.20, 0.30), 10: (0.45, 0.30, 0.25),
    11: (0.45, 0.30, 0.25), 12: (0.50, 0.20, 0.30)
}
CLIMA_OPTS = ["Soleado", "Nublado", "Lluvia"]
CLIMA_IMPACTO = {"Soleado": 1.0, "Nublado": 0.96, "Lluvia": 0.87}
LLUVIA_CONSECUTIVA_PENALTY = 0.02  # -2% extra por cada día seguido de lluvia
LLUVIA_FINDE_PENALTY = 0.06  # -6% adicional si llueve en sábado/domingo

# --- Festivos y ventanas ---
FESTIVOS = {(1, 1), (5, 1), (7, 28), (7, 29), (12, 25)}
VENTANA_FESTIVO = 2  # días antes/después con efecto de viaje
BOOST_PREPOST_FESTIVO = 0.10  # +10% en ventana festiva

# --- Día de pago (quincena / fin de mes) ---
# La demanda sube en días de cobro porque la gente hace más trámites
DIA_PAGO = {1: 0.06, 15: 0.08, 30: 0.05}

# --- Probabilidades de eventos especiales ---
PROB_EVENTO_FESTIVO = 0.35
PROB_EVENTO_FINDE = 0.04
PROB_EVENTO_NORMAL = 0.025

# --- Parámetros de ruido ---
AR_PHI = 0.6       # coeficiente autorregresivo AR(1)
AR_SIGMA = 35      # innovación del AR(1)
RUIDO_MEDICION_STD = 25  # error de medición base
RUIDO_HETERO_FACTOR = 0.015  # componente heteroscedástico
TREND_NOISE_STD = 0.04  # ruido en el incremento de tendencia

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def generar_clima(mes, historial):
    """Clima con persistencia: si llovió ayer, más probable que llueva hoy."""
    p_sol, p_lluvia, p_nublado = CLIMA_PROBS[mes]
    if len(historial) >= 2 and all(c == "Lluvia" for c in historial[-2:]):
        p_lluvia = min(0.95, p_lluvia * 1.8)
    elif historial and historial[-1] == "Lluvia":
        p_lluvia = min(0.90, p_lluvia * 1.4)
    # Rebalancear
    resto = 1.0 - p_lluvia
    p_sol = p_sol / (p_sol + p_nublado) * resto if (p_sol + p_nublado) > 0 else resto / 2
    p_nublado = 1.0 - p_lluvia - p_sol
    return random.choices(CLIMA_OPTS, weights=[p_sol, p_lluvia, p_nublado], k=1)[0]


def contar_lluvia_consecutiva(historial):
    """Días consecutivos de lluvia hasta hoy (sin incluir hoy)."""
    count = 0
    for c in reversed(historial):
        if c == "Lluvia":
            count += 1
        else:
            break
    return count


def generar_evento(festivo, dia_semana):
    """Genera eventos especiales (conciertos, cortes, festividades)."""
    if festivo:
        if random.random() < PROB_EVENTO_FESTIVO:
            return random.randint(200, 600)
    elif dia_semana >= 5 and random.random() < PROB_EVENTO_FINDE:
        return random.randint(100, 400)
    elif random.random() < PROB_EVENTO_NORMAL:
        # Eventos imprevistos: pueden ser positivos (feria) o negativos (corte)
        return random.randint(-200, 300)
    return 0


def es_festivo(fecha):
    return 1 if (fecha.month, fecha.day) in FESTIVOS else 0


# ============================================================
# PRE-CÓMPUTOS
# ============================================================

todas_fechas = [FECHA_INICIO + timedelta(days=i) for i in range(NUM_DIAS)]

# Fechas en ventana de festivo (pre/post viaje)
fechas_ventana_festivo = set()
for i, fecha in enumerate(todas_fechas):
    for offset in range(-VENTANA_FESTIVO, VENTANA_FESTIVO + 1):
        if offset == 0:
            continue
        idx = i + offset
        if 0 <= idx < len(todas_fechas):
            check = todas_fechas[idx]
            if (check.month, check.day) in FESTIVOS:
                fechas_ventana_festivo.add(fecha)
                break


def generate_data(output_csv="demanda_transporte.csv", web_output_csv=None, verbose=True):
    registros = []

    for ruta in RUTAS:

        demanda_base = DEMANDA_BASE[ruta]
        drift = TREND_DRIFT[ruta]

        # --- Tendencia como random walk (crecimiento con perturbaciones) ---
        trend_inc = np.random.normal(drift, TREND_NOISE_STD, NUM_DIAS)
        trend = np.cumsum(trend_inc)

        # --- Ruido AR(1) persistente ---
        ar = np.zeros(NUM_DIAS)
        ar[0] = np.random.normal(0, AR_SIGMA / np.sqrt(1 - AR_PHI**2))
        for i in range(1, NUM_DIAS):
            ar[i] = AR_PHI * ar[i-1] + np.random.normal(0, AR_SIGMA)

        historial_clima = []

        for i in range(NUM_DIAS):

            fecha = todas_fechas[i]
            dia_semana = fecha.weekday()
            mes = fecha.month
            dia_mes = fecha.day

            # ---- 1. Factor semanal con ruido ----
            factor_semana = FACTOR_SEMANA_BASE[dia_semana] * \
                np.random.normal(1, WEEKLY_NOISE_STD)

            # ---- 2. Factor mensual con ruido ----
            factor_mes = FACTOR_MES_BASE[mes] * \
                np.random.normal(1, MONTHLY_NOISE_STD)

            # ---- 3. Clima con persistencia e impacto ----
            clima = generar_clima(mes, historial_clima)
            historial_clima.append(clima)

            factor_clima = CLIMA_IMPACTO[clima]

            if clima == "Lluvia":
                cons_days = contar_lluvia_consecutiva(historial_clima[:-1])
                factor_clima *= (1 - LLUVIA_CONSECUTIVA_PENALTY * cons_days)

            if clima == "Lluvia" and dia_semana >= 5:
                factor_clima *= (1 - LLUVIA_FINDE_PENALTY)

            # ---- 4. Festivo y ventana festiva ----
            festivo = es_festivo(fecha)
            factor_prepost = 1.0 + BOOST_PREPOST_FESTIVO \
                if fecha in fechas_ventana_festivo else 1.0

            # ---- 5. Día de pago ----
            factor_pago = 1.0
            for dia, boost in DIA_PAGO.items():
                if dia_mes == dia:
                    factor_pago = 1.0 + boost * np.random.uniform(0.5, 1.5)
                    break

            # ---- 6. Eventos especiales ----
            evento = generar_evento(festivo, dia_semana)

            # ---- 7. Ruido de medición heteroscedástico ----
            ruido_base = np.random.normal(0, RUIDO_MEDICION_STD)
            ruido_hetero = np.random.normal(0, demanda_base * RUIDO_HETERO_FACTOR)
            ruido_total = ruido_base + ruido_hetero

            # ---- Demanda final ----
            pasajeros = (
                demanda_base
                * factor_semana
                * factor_mes
                * factor_clima
                * factor_prepost
                * factor_pago
                + trend[i]
                + evento
                + ar[i]
                + ruido_total
            )
            pasajeros = max(50, int(round(pasajeros)))

            # ---- Viajes: relación con pasajeros ----
            cap_vehiculo = random.randint(25, 50)
            viajes = max(1, int(round(pasajeros / cap_vehiculo)))

            registros.append({
                "fecha": fecha.strftime("%Y-%m-%d"),
                "ruta": ruta,
                "pasajeros": pasajeros,
                "viajes": viajes,
                "dia_semana": dia_semana,
                "mes": mes,
                "festivo": festivo,
                "clima": clima
            })

    df = pd.DataFrame(registros)

    if verbose:
        print(df.head())
        print(f"\nTotal registros: {len(df)}")

    df.to_csv(output_csv, index=False)

    if verbose:
        print(f"\nDataset sintético guardado como: {output_csv}")

    if web_output_csv:
        os.makedirs(os.path.dirname(web_output_csv), exist_ok=True)
        df.to_csv(web_output_csv, index=False)
        if verbose:
            print(f"Dataset sintético guardado como: {web_output_csv}")

    if verbose:
        print("\n" + "=" * 60)
        print("RESUMEN ESTADÍSTICO")
        print("=" * 60)
        print(df.describe())
        print("\n" + "=" * 60)
        print("PASAJEROS PROMEDIO POR RUTA")
        print("=" * 60)
        print(df.groupby("ruta")["pasajeros"].mean().sort_values(ascending=False))

    return df


if __name__ == "__main__":
    generate_data(web_output_csv="web/public/data/demanda_transporte.csv")
