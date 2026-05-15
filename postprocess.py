#!/usr/bin/env python3
"""
extrae_eventos_headway.py
Versión robusta: registra paradas en orden lógico, calcula el intervalo (headway)
entre buses consecutivos en cada parada y exporta el headway medio por parada.
"""

import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# ================== CONFIGURACIÓN ==================
PARADES_FILE       = "parades.json"
MAX_ORDRE_JUMP     = 5
MIN_DIFF_VIAJE     = 45
RESTART_JUMP       = 10
MAX_ETA_MINUTES    = 60
INPUT_PATH         = "data/arrivals_2026-05-02_8_2.csv"
OUTPUT_EVENTOS_DIR = "eventos_nuevos/"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_parades(file_path: str):
    path = Path(file_path)
    if not path.exists():
        log.warning("No se encuentra el archivo de paradas: %s", file_path)
        return {}
    if path.suffix.lower() == '.json':
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        mapping = {"Anada": "arribada", "Tornada": "tornada"}
        res = {}
        for s_csv, s_json in mapping.items():
            if "itinerarios" in data and s_json in data["itinerarios"]:
                res[s_csv] = {p["nombre"]: p["orden"] for p in data["itinerarios"][s_json]["paradas"]}
        return res
    return {}


def extract_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
    df["temps_arribada_dt"] = pd.to_datetime(df["temps_arribada"], unit='ms', utc=True)
    df["temps_arribada_min"] = (df["temps_arribada_dt"] - df["captured_at"]).dt.total_seconds() / 60.0

    df = df[(df["temps_arribada_min"] >= 0) & (df["temps_arribada_min"] <= MAX_ETA_MINUTES)].copy()

    df = df.sort_values(["id_bus", "sentit", "codi_parada", "captured_at"]).reset_index(drop=True)
    df["time_gap"] = df.groupby(["id_bus", "sentit", "codi_parada"])["captured_at"].diff().dt.total_seconds() / 60.0
    df["trip_group"] = (df["time_gap"] > MIN_DIFF_VIAJE).cumsum()

    idx_best = df.groupby(["id_bus", "sentit", "codi_parada", "trip_group"])["temps_arribada_min"].idxmin()
    df_best = df.loc[idx_best].copy()
    df_best["timestamp_paso"] = df_best["captured_at"] + pd.to_timedelta(df_best["temps_arribada_min"], unit='m')

    df_best = df_best.sort_values(["id_bus", "sentit", "timestamp_paso"]).reset_index(drop=True)

    eventos = []
    bus_state = {}

    for _, row in df_best.iterrows():
        bus_key = (row["id_bus"], row["sentit"])
        ts_paso = row["timestamp_paso"]
        ordre_act = row["ordre"]

        if bus_key not in bus_state:
            bus_state[bus_key] = {"last_ordre": ordre_act, "last_ts": ts_paso, "viaje_n": 1}
            row_to_add = row.copy()
            row_to_add["viaje_n"] = 1
            eventos.append(row_to_add)
            continue

        state = bus_state[bus_key]
        last_ordre = state["last_ordre"]
        last_ts = state["last_ts"]
        viaje_n = state["viaje_n"]
        tiempo_diff = (ts_paso - last_ts).total_seconds() / 60.0

        es_nuevo_viaje = (tiempo_diff > MIN_DIFF_VIAJE) or (
            ordre_act < last_ordre and (last_ordre - ordre_act) >= RESTART_JUMP
        )

        if es_nuevo_viaje:
            viaje_n += 1
            state.update({"viaje_n": viaje_n, "last_ordre": ordre_act, "last_ts": ts_paso})
            row_to_add = row.copy()
            row_to_add["viaje_n"] = viaje_n
            eventos.append(row_to_add)

        elif ordre_act > last_ordre:
            if (ordre_act - last_ordre) <= MAX_ORDRE_JUMP:
                state.update({"last_ordre": ordre_act, "last_ts": ts_paso})
                row_to_add = row.copy()
                row_to_add["viaje_n"] = viaje_n
                eventos.append(row_to_add)

    if not eventos:
        return pd.DataFrame()

    df_ev = pd.DataFrame(eventos)

    # Calcular missings
    df_ev["ordre_faltantes"] = ""
    for (bus, sentit, v_id), grp in df_ev.groupby(["id_bus", "sentit", "viaje_n"]):
        presentes = sorted(grp["ordre"].dropna().unique())
        if presentes:
            rango_completo = set(range(int(presentes[0]), int(presentes[-1]) + 1))
            faltantes = rango_completo - set(presentes)
            if faltantes:
                df_ev.loc[grp.index, "ordre_faltantes"] = ",".join(str(o) for o in sorted(faltantes))

    return df_ev


def calculate_missings(df: pd.DataFrame) -> None:
    missing = 0
    total_registros = len(df)
    for (id_bus, viaje) in df.groupby(["id_bus", "viaje_n"]):
        if viaje["ordre_faltantes"].iloc[0] != "":
            log.info(
                "Bus %s (Viaje %d) - Paradas faltantes: %s",
                id_bus, viaje["viaje_n"].iloc[0], viaje["ordre_faltantes"].iloc[0]
            )
            missing += len(viaje["ordre_faltantes"].iloc[0].split(","))

    print(missing)
    print(f"Porcentaje de paradas faltantes: {missing / total_registros * 100:.2f}%")


def calculate_mean_headway(df_ev: pd.DataFrame):
    """
    Calcula el headway entre buses, eliminando el 5% de las muestras 
    extremas (outliers) antes de calcular estadísticas.
    """
    df_sorted = df_ev.sort_values(["sentit", "codi_parada", "timestamp_paso"]).copy()

    df_sorted["headway_min"] = (
        df_sorted.groupby(["sentit", "codi_parada"])["timestamp_paso"]
        .diff()
        .dt.total_seconds() / 60.0
    )

    Q1 = df_sorted["headway_min"].quantile(0.25)
    Q3 = df_sorted["headway_min"].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = max(1.0, Q1 - 1.5 * IQR) # No permitimos menos de 1 min #TODO Fixear 
    upper_bound = Q3 + 1.5 * IQR

    df_filtered = df_sorted[
        (df_sorted["headway_min"] >= lower_bound) & 
        (df_sorted["headway_min"] <= upper_bound)
    ].copy()

    mean_hw = (
        df_filtered.groupby(["sentit", "codi_parada", "nom_parada", "ordre"])
        .agg(
            headway_medio_min=("headway_min", "mean"),
            headway_mediana_min=("headway_min", "median"),
            headway_std_min=("headway_min", "std"),
            num_pasos_utiles=("headway_min", "count"),
        )
        .round(2)
        .reset_index()
        .sort_values(["sentit", "ordre"])
    )

    return mean_hw, df_sorted

import matplotlib.pyplot as plt
import seaborn as sns

def plot_headway_analysis(df_ev_con_hw):
    # Filtrar NaNs para el gráfico
    data_plot = df_ev_con_hw.dropna(subset=['headway_min'])
    
    # Configurar estilo
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Histograma con KDE
    sns.histplot(data=data_plot, x="headway_min", bins=50, kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title('Distribución Global de Headways', fontsize=14)
    axes[0].set_xlabel('Minutos entre buses')
    axes[0].set_ylabel('Frecuencia')

    # 2. Boxplot para ver Outliers por Sentido
    sns.boxplot(data=data_plot, x="sentit", y="headway_min", ax=axes[1], palette="Set2")
    axes[1].set_title('Dispersión y Outliers por Sentido', fontsize=14)
    axes[1].set_ylabel('Headway (min)')
    axes[1].set_xlabel('Sentido')

    plt.tight_layout()
    plt.show()

def process_file(input_path, output_path, parades_map):
    log.info("Procesando: %s", input_path.name)
    df = pd.read_csv(input_path, dtype={"id_bus": str, "codi_parada": str})
    df = df.drop_duplicates()

    if parades_map:
        df["ordre"] = df.apply(
            lambda r: parades_map.get(r["sentit"], {}).get(r["nom_parada"], r["ordre"]), axis=1
        )

    df = df.dropna(subset=["id_bus", "sentit", "codi_parada", "temps_arribada", "captured_at"])
    df_eventos = extract_events(df)

    calculate_missings(df_eventos)

    if not df_eventos.empty:
        mean_hw_df, df_eventos_con_hw = calculate_mean_headway(df_eventos)

        plot_headway_analysis(df_eventos_con_hw)
        log.info("Headway medio global: %.2f min", mean_hw_df["headway_medio_min"].mean())
        log.info("\n%s", mean_hw_df.to_string(index=False))

        log.info("Std media del headway: %.2f min", mean_hw_df["headway_std_min"].mean())


        hw_output_path = output_path.parent / f"{output_path.stem}_headway_medio.csv"
        mean_hw_df.to_csv(hw_output_path, index=False)
        log.info("Headway medio guardado en: %s", hw_output_path)

        df_eventos_con_hw["hora_paso"] = df_eventos_con_hw["timestamp_paso"].dt.strftime("%H:%M:%S")
        df_eventos_con_hw["min_paso"] = (
            df_eventos_con_hw["timestamp_paso"].dt.minute
            + df_eventos_con_hw["timestamp_paso"].dt.second / 60.0
        ).round(2)
        
        df_eventos_con_hw["headway_sec"] = (df_eventos_con_hw["headway_min"] * 60).round(2)
        
        initial_rows = len(df_eventos_con_hw)
        df_eventos_con_hw = df_eventos_con_hw[
            (df_eventos_con_hw["headway_sec"].isna()) | (df_eventos_con_hw["headway_sec"] >= 5)
        ].copy()
        removed_rows = initial_rows - len(df_eventos_con_hw)
        log.info("Eliminadas %d filas con headway < 5 segundos", removed_rows)

        df_eventos_con_hw.sort_values(["sentit", "ordre", "timestamp_paso"], inplace=True)

        cols_final = [
            "id_bus", "viaje_n", "nom_linia", "sentit", "ordre",
            "nom_parada", "codi_parada", "hora_paso", "min_paso",
            "headway_min", "headway_sec", "ordre_faltantes",
        ]
        cols_final = [c for c in cols_final if c in df_eventos_con_hw.columns]
        '''
        df_eventos_con_hw[cols_final].to_csv(output_path, index=False)
        '''
        log.info("Eventos guardados en: %s (%d filas)", output_path, len(df_eventos_con_hw))
    else:
        log.warning("No se detectaron eventos válidos.")


def main():
    p_map = load_parades(PARADES_FILE)
    out_dir = Path(OUTPUT_EVENTOS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    in_path = Path(INPUT_PATH)

    if in_path.is_file():
        process_file(in_path, out_dir / f"{in_path.stem}_limpio_3.csv", p_map)
    else:
        for f in in_path.glob("*.csv"):
            if "limpio" not in f.name:
                process_file(f, out_dir / f"{f.stem}_limpio_3.csv", p_map)


if __name__ == "__main__":
    main()
    