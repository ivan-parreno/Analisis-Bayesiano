#!/usr/bin/env python3
"""
extrae_eventos_headway.py
Versión flexible y robusta: permite elegir entre procesar archivos individuales
o consolidar carpetas, corrigiendo el ruido de paradas faltantes en terminales.
"""

import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================== CONFIGURACIÓN ==================
PARADES_FILE          = "parades.json"
MAX_ORDRE_JUMP        = 5
MIN_DIFF_VIAJE        = 45
RESTART_JUMP          = 10
MAX_ETA_MINUTES       = 60
INPUT_PATH            = "raw/data16"
OUTPUT_EVENTOS_DIR    = "processed/eventos/"
MIN_HEADWAY_SEG       = 5   # Eliminar headways menores a 5 segundos (ruido)

# --- VARIABLE DE SELECCIÓN DE MODO ---
# True  => Combina todos los archivos en un único reporte y gráfica global.
# False => Procesa cada archivo por separado (un reporte, limpio y gráfica por .csv).
CONSOLIDATE_ALL_FILES = True  
# ===================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_parades(file_path: str):
    """Carga el JSON de paradas y devuelve un dict {sentido: {nombre_parada: ordre}}."""
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


def get_ordenes_esperados(df, parades_map):
    """Devuelve un diccionario {sentido: conjunto de órdenes completos}."""
    ordenes_por_sentido = {}
    for sentido in df["sentit"].unique():
        if sentido in parades_map and parades_map[sentido]:
            ordenes = set(parades_map[sentido].values())
            ordenes_completos = set(range(1, max(ordenes) + 1))
            ordenes_por_sentido[sentido] = ordenes_completos
        else:
            max_ordre = df[df["sentit"] == sentido]["ordre"].max()
            if pd.notna(max_ordre):
                ordenes_por_sentido[sentido] = set(range(1, int(max_ordre) + 1))
            else:
                ordenes_por_sentido[sentido] = set()
    return ordenes_por_sentido


def extract_events(df: pd.DataFrame, ordenes_esperados: dict) -> pd.DataFrame:
    """Extrae eventos de paso por parada, asigna número de viaje y detecta paradas faltantes."""
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

    # --- CORRECCIÓN AQUÍ: Evitar el ruido de la última parada de la línea ---
    df_ev["ordre_faltantes"] = ""
    for (bus, sentit, v_id), grp in df_ev.groupby(["id_bus", "sentit", "viaje_n"]):
        ordenes_presentes = set(grp["ordre"].dropna().astype(int))
        
        if sentit in ordenes_esperados and ordenes_esperados[sentit]:
            # Obtenemos la lista esperada quitando el máximo (la parada terminal)
            max_esperado = max(ordenes_esperados[sentit])
            ordenes_completos = ordenes_esperados[sentit] - {max_esperado}
            
            faltantes = ordenes_completos - ordenes_presentes
            if faltantes:
                df_ev.loc[grp.index, "ordre_faltantes"] = ",".join(str(o) for o in sorted(faltantes))
        else:
            # Si no hay JSON, calculamos dinámicamente pero ignorando también el máximo detectado
            if len(ordenes_presentes) > 1:
                min_o, max_o = min(ordenes_presentes), max(ordenes_presentes)
                ordenes_completos = set(range(min_o, max_o)) # El range para en max_o - 1
                faltantes = ordenes_completos - ordenes_presentes
                if faltantes:
                    df_ev.loc[grp.index, "ordre_faltantes"] = ",".join(str(o) for o in sorted(faltantes))

    return df_ev


def calculate_mean_headway(df_ev: pd.DataFrame):
    """Calcula el headway agrupando por origen de archivo para evitar saltos temporales falsos."""
    df_sorted = df_ev.sort_values(["origen_archivo", "sentit", "codi_parada", "timestamp_paso"]).copy()

    df_sorted["headway_min"] = (
        df_sorted.groupby(["origen_archivo", "sentit", "codi_parada"])["timestamp_paso"]
        .diff()
        .dt.total_seconds() / 60.0
    )

    df_valid = df_sorted.dropna(subset=["headway_min"])
    if df_valid.empty:
        return pd.DataFrame(), df_sorted

    Q1 = df_valid["headway_min"].quantile(0.25)
    Q3 = df_valid["headway_min"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(1.0, Q1 - 1.5 * IQR)
    upper_bound = Q3 + 1.5 * IQR

    df_filtered = df_sorted[
        (df_sorted["headway_min"].isna()) | 
        ((df_sorted["headway_min"] >= lower_bound) & (df_sorted["headway_min"] <= upper_bound))
    ].copy()

    mean_hw = (
        df_filtered.dropna(subset=["headway_min"])
        .groupby(["sentit", "codi_parada", "nom_parada", "ordre"])
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

    return mean_hw, df_filtered


def plot_headway_analysis(df_ev_con_hw, title_suffix="", save_figures=True):
    data_plot = df_ev_con_hw.dropna(subset=['headway_min'])
    if data_plot.empty:
        log.warning("No hay datos para graficar headway.")
        return
    if not save_figures:
        return
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(data=data_plot, x="headway_min", bins=50, kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title(f'Distribución de Headways {title_suffix}', fontsize=14)
    axes[0].set_xlabel('Minutos entre buses')
    axes[0].set_ylabel('Frecuencia')
    sns.boxplot(data=data_plot, x="sentit", y="headway_min", ax=axes[1], palette="Set2")
    axes[1].set_title(f'Dispersión por Sentido {title_suffix}', fontsize=14)
    axes[1].set_ylabel('Headway (min)')
    axes[1].set_xlabel('Sentido')
    plt.tight_layout()
    plt.show()


def process_single_file(file_path: Path, parades_map: dict) -> pd.DataFrame:
    """Procesa la estructura base de un archivo y devuelve su DataFrame de eventos."""
    log.info("Extrayendo eventos de: %s", file_path.name)
    df = pd.read_csv(file_path, dtype={"id_bus": str, "codi_parada": str})
    df = df.drop_duplicates()

    if parades_map:
        df["ordre"] = df.apply(
            lambda r: parades_map.get(r["sentit"], {}).get(r["nom_parada"], r["ordre"]), axis=1
        )

    df = df.dropna(subset=["id_bus", "sentit", "codi_parada", "temps_arribada", "captured_at"])
    ordenes_esperados = get_ordenes_esperados(df, parades_map)
    
    df_eventos = extract_events(df, ordenes_esperados)
    if not df_eventos.empty:
        df_eventos["origen_archivo"] = file_path.stem
    return df_eventos


def main(input_path=None, output_dir=None, save_figures=True, date_filter=None):
    input_path = input_path or INPUT_PATH
    output_dir = output_dir or OUTPUT_EVENTOS_DIR

    p_map = load_parades(PARADES_FILE)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    in_path = Path(input_path)

    files_to_process = []
    if in_path.is_file() and in_path.suffix.lower() == '.csv':
        files_to_process.append(in_path)
    elif in_path.is_dir():
        files_to_process = sorted(
            f for f in in_path.glob("*.csv") if "limpio" not in f.name
        )

    if date_filter:
        files_to_process = [
            f for f in files_to_process if date_filter in f.name
        ]

    if not files_to_process:
        log.error("No se encontraron archivos .csv válidos en: %s", input_path)
        return []

    generated = []

    # ========================================================
    # MODO 1: CONSOLIDAR TODO EL DIRECTORIO
    # ========================================================
    if CONSOLIDATE_ALL_FILES:
        log.info("--- MODO GLOBAL: Consolidando %d archivos ---", len(files_to_process))
        list_all_events = []
        for f in files_to_process:
            df_ev_file = process_single_file(f, p_map)
            if not df_ev_file.empty:
                list_all_events.append(df_ev_file)

        if not list_all_events:
            log.warning("No se generaron eventos en ningún archivo.")
            return []

        df_global_events = pd.concat(list_all_events, ignore_index=True)
        mean_hw_df, df_global_con_hw = calculate_mean_headway(df_global_events)

        if mean_hw_df.empty:
            return []

        log.info("=== REPORTE CONSOLIDADO DE LA CARPETA ===")
        log.info("Headway medio global: %.2f min", mean_hw_df["headway_medio_min"].mean())
        log.info("\n%s", mean_hw_df.to_string(index=False))

        global_hw_output = out_dir / "reporte_global_headway_medio.csv"
        mean_hw_df.to_csv(global_hw_output, index=False)
        log.info("Reporte global guardado en: %s", global_hw_output)

        df_global_con_hw["hora_paso"] = df_global_con_hw["timestamp_paso"].dt.strftime("%H:%M:%S")
        df_global_con_hw["min_paso"] = (df_global_con_hw["timestamp_paso"].dt.minute + df_global_con_hw["timestamp_paso"].dt.second / 60.0).round(2)
        df_global_con_hw["headway_sec"] = (df_global_con_hw["headway_min"] * 60).round(2)

        df_global_con_hw = df_global_con_hw[(df_global_con_hw["headway_sec"].isna()) | (df_global_con_hw["headway_sec"] >= MIN_HEADWAY_SEG)].copy()

        cols_final = ["id_bus", "viaje_n", "nom_linia", "sentit", "ordre", "nom_parada", "codi_parada", "hora_paso", "min_paso", "headway_min", "headway_sec", "ordre_faltantes"]
        cols_final = [c for c in cols_final if c in df_global_con_hw.columns]

        for origen, grp in df_global_con_hw.groupby("origen_archivo"):
            out_file_path = out_dir / f"{origen}_limpio_3.csv"
            grp.sort_values(["sentit", "ordre", "timestamp_paso"])[cols_final].to_csv(out_file_path, index=False)
            generated.append(out_file_path)

        plot_headway_analysis(df_global_con_hw, title_suffix="(Global Carpeta)", save_figures=save_figures)

    # ========================================================
    # MODO 2: PROCESAR CADA ARCHIVO INDEPENDIENTEMENTE
    # ========================================================
    else:
        log.info("--- MODO INDIVIDUAL: Procesando archivos por separado ---")
        for f in files_to_process:
            df_ev_file = process_single_file(f, p_map)
            if df_ev_file.empty:
                continue

            mean_hw_df, df_file_con_hw = calculate_mean_headway(df_ev_file)
            if mean_hw_df.empty:
                continue

            log.info("=== REPORTE INDIVIDUAL: %s ===", f.name)
            log.info("Headway medio de este archivo: %.2f min", mean_hw_df["headway_medio_min"].mean())
            
            hw_output_path = out_dir / f"{f.stem}_headway_medio.csv"
            mean_hw_df.to_csv(hw_output_path, index=False)

            df_file_con_hw["hora_paso"] = df_file_con_hw["timestamp_paso"].dt.strftime("%H:%M:%S")
            df_file_con_hw["min_paso"] = (df_file_con_hw["timestamp_paso"].dt.minute + df_file_con_hw["timestamp_paso"].dt.second / 60.0).round(2)
            df_file_con_hw["headway_sec"] = (df_file_con_hw["headway_min"] * 60).round(2)

            df_file_con_hw = df_file_con_hw[(df_file_con_hw["headway_sec"].isna()) | (df_file_con_hw["headway_sec"] >= MIN_HEADWAY_SEG)].copy()

            cols_final = ["id_bus", "viaje_n", "nom_linia", "sentit", "ordre", "nom_parada", "codi_parada", "hora_paso", "min_paso", "headway_min", "headway_sec", "ordre_faltantes"]
            cols_final = [c for c in cols_final if c in df_file_con_hw.columns]

            out_file_path = out_dir / f"{f.stem}_limpio_3.csv"
            df_file_con_hw.sort_values(["sentit", "ordre", "timestamp_paso"])[cols_final].to_csv(out_file_path, index=False)
            generated.append(out_file_path)

            plot_headway_analysis(df_file_con_hw, title_suffix=f"({f.stem})", save_figures=save_figures)

    return generated


def parse_args():
    parser = argparse.ArgumentParser(description="Procesa llegadas brutas y genera eventos limpios.")
    parser.add_argument("--input", default=INPUT_PATH, help="Archivo o carpeta CSV de entrada")
    parser.add_argument("--output", default=OUTPUT_EVENTOS_DIR, help="Carpeta de salida para *_limpio_3.csv")
    parser.add_argument("--date", default=None, help="Filtrar por fecha (YYYY-MM-DD) en el nombre del archivo")
    parser.add_argument("--no-figures", action="store_true", help="No mostrar gráficos de headway")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        input_path=args.input,
        output_dir=args.output,
        save_figures=not args.no_figures,
        date_filter=args.date,
    )
