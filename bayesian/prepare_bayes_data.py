#!/usr/bin/env python3
"""Prepara datasets para Modelo 1 (llegadas) y Modelo 2 (pares)."""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    EVENTOS_DIR,
    HEADWAYS_DIR,
    HW_MAX,
    HW_MIN,
    LINEA,
    SENTIDO,
    Y_SHIFT,
    hora_a_franja,
    tiempo_str_a_minutos,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_hours import calcular_headways_parada, cargar_datos  # noqa: E402


def fecha_desde_nombre(path: Path) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    if not m:
        raise ValueError(f"No se encontró fecha en {path.name}")
    return m.group(1)


def build_headways_stop(
    eventos_dir: Path, linea: str, sentido: str, fecha_filter: str | None = None
) -> pd.DataFrame:
    frames = []
    for f in sorted(eventos_dir.glob("*_limpio_3.csv")):
        fecha = fecha_desde_nombre(f)
        if fecha_filter and fecha != fecha_filter:
            continue
        df = cargar_datos(str(f))
        mask = (df["nom_linia"] == linea) & (df["sentit"] == sentido) & (df["ordre"] >= 1)
        df_f = df[mask].copy()
        if df_f.empty:
            continue
        weekday = pd.Timestamp(fecha).dayofweek
        for cod_parada, grupo in df_f.groupby("codi_parada"):
            hw = calcular_headways_parada(grupo, max_headway_min=HW_MAX)
            if hw.empty:
                continue
            ordre = int(grupo["ordre"].iloc[0])
            hw = hw.copy()
            hw["codi_parada"] = cod_parada
            hw["ordre"] = ordre
            hw["fecha"] = fecha
            hw["weekday"] = weekday
            hw["franja"] = hw["hora"].astype(int).apply(hora_a_franja)
            hw["headway_min"] = hw["headway_min"]
            hw = hw[(hw["headway_min"] >= HW_MIN) & (hw["headway_min"] <= HW_MAX)]
            if hw.empty:
                continue
            frames.append(hw)
    if not frames:
        raise RuntimeError("No se generaron headways de parada.")
    out = pd.concat(frames, ignore_index=True)
    out["y_shifted"] = out["headway_min"] - Y_SHIFT
    out = out[out["y_shifted"] > 0]
    return out


def _load_headways_pares_files(
    headways_dir: Path, linea: str, sentido: str, fecha_filter: str | None = None
) -> list[pd.DataFrame]:
    """Carga CSVs de pares con fecha, weekday, franja y hora_min."""
    pattern = re.compile(
        rf"headways_pares3_{re.escape(linea)}_{re.escape(sentido)}_(\d{{4}}-\d{{2}}-\d{{2}})\.csv"
    )
    frames = []
    for f in sorted(headways_dir.glob("headways_pares3_*.csv")):
        m = pattern.match(f.name)
        if not m:
            continue
        fecha = m.group(1)
        if fecha_filter and fecha != fecha_filter:
            continue
        weekday = pd.Timestamp(fecha).dayofweek
        df = pd.read_csv(f)
        if df.empty:
            continue
        df["fecha"] = fecha
        df["weekday"] = weekday
        if "hora_delante_str" in df.columns:
            df["hora_min"] = df["hora_delante_str"].apply(tiempo_str_a_minutos)
        elif "franja_hora" in df.columns:
            df["hora_min"] = df["franja_hora"].astype(float) * 60
        else:
            df["hora_min"] = np.nan
        # franja_hora en el CSV es hora reloj (0-23), NO el código 0-16 del modelo
        df["franja"] = df["hora_min"].apply(
            lambda x: hora_a_franja(int(x // 60)) if pd.notna(x) else np.nan
        )
        df = df.dropna(subset=["franja", "headway_pair", "ordre"])
        df = df[(df["headway_pair"] >= HW_MIN) & (df["headway_pair"] <= HW_MAX)]
        frames.append(df)
    return frames


def build_headways_stop_from_pairs(
    headways_dir: Path, linea: str, sentido: str, fecha_filter: str | None = None
) -> pd.DataFrame:
    """
    Headway por parada desde pares delante→detrás (headway_pair).
    Cada fila es el gap real entre los dos viajes emparejados en esa parada.
    """
    frames = []
    for df in _load_headways_pares_files(headways_dir, linea, sentido, fecha_filter):
        hw = df.copy()
        hw["headway_min"] = hw["headway_pair"]
        hw["hora"] = hw["hora_min"] / 60.0
        frames.append(
            hw[["hora", "headway_min", "codi_parada", "ordre", "fecha", "weekday", "franja"]]
        )
    if not frames:
        raise RuntimeError("No se generaron headways desde pares.")
    out = pd.concat(frames, ignore_index=True)
    out["y_shifted"] = out["headway_min"] - Y_SHIFT
    out = out[out["y_shifted"] > 0]
    return out


def build_headways_pair(
    headways_dir: Path, linea: str, sentido: str, fecha_filter: str | None = None
) -> pd.DataFrame:
    frames = _load_headways_pares_files(headways_dir, linea, sentido, fecha_filter)
    if not frames:
        raise RuntimeError("No se generaron headways de pares.")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(
        ["codi_parada", "fecha", "hora_min", "par_id", "ordre"]
    )
    # Lag espacial (mismo par, ordre-1)
    out["ordre_ant"] = out["ordre"] - 1
    lag_esp = out[["par_id", "ordre", "headway_pair"]].rename(
        columns={"ordre": "ordre_ant", "headway_pair": "hw_lag_esp"}
    )
    out = out.merge(lag_esp, on=["par_id", "ordre_ant"], how="left")
    # Lag temporal (par anterior en misma parada)
    out["hw_lag_temp"] = out.groupby("codi_parada")["headway_pair"].shift(1)
    out["existe_lag_temp"] = out["hw_lag_temp"].notna().astype(int)
    out = out.dropna(subset=["hw_lag_esp"])
    out["y_shifted"] = out["headway_pair"] - Y_SHIFT
    out["y_lag_esp_shifted"] = out["hw_lag_esp"] - Y_SHIFT
    out["y_lag_temp_shifted"] = out["hw_lag_temp"].fillna(0) - Y_SHIFT
    out = out[out["y_shifted"] > 0]
    out = out[out["y_lag_esp_shifted"] > 0]
    return out


def encode_indices(df_stop: pd.DataFrame, df_pair: pd.DataFrame) -> dict:
    paradas = sorted(
        set(df_stop["codi_parada"].astype(str)) | set(df_pair["codi_parada"].astype(str))
    )
    ordres = sorted(
        set(df_stop["ordre"].astype(int)) | set(df_pair["ordre"].astype(int))
    )
    par_id_list = sorted(df_pair["par_id"].astype(str).unique())

    maps = {
        "parada": {p: i for i, p in enumerate(paradas)},
        "ordre": {int(o): i for i, o in enumerate(ordres)},
        "franja": {h: h for h in range(17)},
        "weekday": {d: d for d in range(7)},
        "par_id": {p: i for i, p in enumerate(par_id_list)},
        "ordre_to_parada": {},
    }
    for _, row in df_stop.drop_duplicates(["ordre", "codi_parada"]).iterrows():
        maps["ordre_to_parada"][str(int(row["ordre"]))] = str(row["codi_parada"])

    return maps


def apply_maps(df_stop: pd.DataFrame, df_pair: pd.DataFrame, maps: dict):
    df_stop = df_stop.copy()
    df_pair = df_pair.copy()
    mp = maps["parada"]
    df_stop["s"] = df_stop["codi_parada"].astype(str).map(mp)
    df_stop["h"] = df_stop["franja"].astype(int)
    df_stop["d"] = df_stop["weekday"].astype(int)
    df_stop = df_stop.dropna(subset=["s", "h", "d"])

    if df_pair.empty:
        return df_stop, df_pair

    df_pair["s"] = df_pair["codi_parada"].astype(str).map(mp)
    df_pair["o"] = df_pair["ordre"].astype(int).map(maps["ordre"])
    df_pair["h"] = df_pair["franja"].astype(int)
    df_pair["d"] = df_pair["weekday"].astype(int)
    df_pair["par_idx"] = df_pair["par_id"].astype(str).map(maps["par_id"])

    # índice parada anterior por ordre
    ordre_to_s = {}
    for o_str, cod in maps["ordre_to_parada"].items():
        ordre_to_s[int(o_str)] = mp[cod]
    df_pair["o_ant"] = df_pair["ordre"].astype(int) - 1
    df_pair["s_ant"] = df_pair["o_ant"].map(ordre_to_s)

    df_pair = df_pair.dropna(subset=["s", "o", "h", "d", "par_idx", "s_ant"])
    return df_stop, df_pair


def subsample_df(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if n <= 0 or n >= len(df):
        return df
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Preparar datos bayesianos.")
    parser.add_argument("--eventos", default=str(EVENTOS_DIR))
    parser.add_argument("--headways", default=str(HEADWAYS_DIR))
    parser.add_argument("--output", default=None, help=f"Carpeta salida (default: {DATA_DIR} o data/FECHA con --date)")
    parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        help="Procesar solo un día (filtra llegadas y pares por fecha).",
    )
    parser.add_argument(
        "--source",
        choices=["pairs", "arrivals"],
        default="pairs",
        help="Origen de headways_stop: 'pairs' (delante→detrás, recomendado) o "
        "'arrivals' (gap cronológico entre buses en la parada).",
    )
    parser.add_argument(
        "--stop-only",
        action="store_true",
        help="Solo generar headways_stop (suficiente para Modelo 1).",
    )
    parser.add_argument(
        "--subsample-stop",
        type=int,
        default=0,
        help="Máx. filas en headways_stop (0=todas). Útil para pruebas rápidas.",
    )
    parser.add_argument(
        "--subsample-pair",
        type=int,
        default=0,
        help="Máx. filas en headways_pair (0=todas). Útil para pruebas rápidas.",
    )
    args = parser.parse_args()

    if args.output:
        out_dir = Path(args.output)
    elif args.date:
        out_dir = DATA_DIR / args.date
    else:
        out_dir = DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    label = f" (día {args.date})" if args.date else ""
    src_label = "pares delante→detrás" if args.source == "pairs" else "llegadas cronológicas"
    print(f"Construyendo headways_stop{label} [{src_label}]...")
    if args.source == "pairs":
        df_stop = build_headways_stop_from_pairs(
            Path(args.headways), LINEA, SENTIDO, fecha_filter=args.date
        )
    else:
        df_stop = build_headways_stop(
            Path(args.eventos), LINEA, SENTIDO, fecha_filter=args.date
        )
    print(f"  {len(df_stop)} observaciones")

    if args.stop_only:
        df_pair = pd.DataFrame(columns=["codi_parada", "ordre", "par_id"])
        print("  (--stop-only: omitiendo headways_pair)")
    else:
        print(f"Construyendo headways_pair{label}...")
        df_pair = build_headways_pair(
            Path(args.headways), LINEA, SENTIDO, fecha_filter=args.date
        )
        print(f"  {len(df_pair)} observaciones de pares")

    maps = encode_indices(df_stop, df_pair)
    df_stop, df_pair = apply_maps(df_stop, df_pair, maps)

    df_stop = subsample_df(df_stop, args.subsample_stop)
    df_pair = subsample_df(df_pair, args.subsample_pair)

    df_stop.to_csv(out_dir / "headways_stop.csv", index=False)
    if not args.stop_only:
        df_pair.to_csv(out_dir / "headways_pair.csv", index=False)
    with open(out_dir / "maps.json", "w", encoding="utf-8") as f:
        json.dump(maps, f, indent=2)

    print(f"\nGuardado en {out_dir}/")
    n_pares = len(maps["par_id"])
    print(f"  Paradas: {len(maps['parada'])}, Pares únicos: {n_pares}", end="")
    if args.stop_only:
        print(" (0 es normal con --stop-only: no se exporta headways_pair para M2)")
    else:
        print()
    if args.date:
        print(f"\nModelo 1 (R + Stan) con un día:")
        print(f"  cd R && Rscript fit_model1.R --data {out_dir} --iter 400 --warmup 200")


if __name__ == "__main__":
    main()
