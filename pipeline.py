#!/usr/bin/env python3
"""Pipeline de limpieza: brutos → llegadas (*_limpio_3.csv) + headways_pares."""

import argparse
import re
import sys
from pathlib import Path

from plot_hours import (
    LINEA_SELECCIONADA,
    OUTPUT_HEADWAYS_DIR,
    SENTIDO_SELECCIONADO,
    export_headways_pares,
    extract_date_from_limpio,
)
from postprocess import OUTPUT_EVENTOS_DIR, main as run_postprocess

DEFAULT_RAW = "raw/data16"
DEFAULT_EVENTOS = OUTPUT_EVENTOS_DIR
DEFAULT_HEADWAYS = OUTPUT_HEADWAYS_DIR


def headways_path(eventos_dir, linea, sentido, fecha):
    return Path(eventos_dir).parent / "headways" / f"headways_pares3_{linea}_{sentido}_{fecha}.csv"


def limpio_path_for_raw(raw_file, eventos_dir):
    return Path(eventos_dir) / f"{raw_file.stem}_limpio_3.csv"


def collect_limpio_files(eventos_dir, date_filter=None):
    files = sorted(Path(eventos_dir).glob("*_limpio_3.csv"))
    if date_filter:
        files = [f for f in files if date_filter in f.name]
    return files


def process_headways(limpio_files, linea, sentido, headways_dir):
    generated = []
    for limpio in limpio_files:
        fecha = extract_date_from_limpio(limpio)
        expected = Path(headways_dir) / f"headways_pares3_{linea}_{sentido}_{fecha}.csv"
        if expected.exists():
            print(f"Headways ya existen: {expected.name}")
            generated.append(expected)
            continue
        out = export_headways_pares(
            limpio,
            linea=linea,
            sentido=sentido,
            output_dir=headways_dir,
            save_figures=False,
        )
        if out:
            generated.append(out)
    return generated


def validate_pairs(raw_files, eventos_dir, headways_dir, linea, sentido):
    errors = []
    for raw in raw_files:
        fecha_match = re.search(r"(\d{4}-\d{2}-\d{2})", raw.name)
        if not fecha_match:
            continue
        fecha = fecha_match.group(1)
        limpio = Path(eventos_dir) / f"{raw.stem}_limpio_3.csv"
        headways = Path(headways_dir) / f"headways_pares3_{linea}_{sentido}_{fecha}.csv"
        if not limpio.exists():
            errors.append(f"Falta llegadas: {limpio.name}")
        if not headways.exists():
            errors.append(f"Falta headways_pares: {headways.name}")
    return errors


def validate_limpios(limpio_files, headways_dir, linea, sentido):
    errors = []
    for limpio in limpio_files:
        fecha = extract_date_from_limpio(limpio)
        if not fecha:
            errors.append(f"No se pudo extraer fecha de {limpio.name}")
            continue
        headways = Path(headways_dir) / f"headways_pares3_{linea}_{sentido}_{fecha}.csv"
        if not headways.exists():
            errors.append(f"Falta headways_pares: {headways.name}")
    return errors


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de limpieza de datos de autobuses.")
    parser.add_argument("--input", default=None, help="Carpeta o archivo raw CSV (brutos)")
    parser.add_argument("--eventos", default=DEFAULT_EVENTOS, help="Carpeta de salida para *_limpio_3.csv")
    parser.add_argument("--headways-dir", default=DEFAULT_HEADWAYS, help="Carpeta de salida para headways_pares")
    parser.add_argument("--linea", default=LINEA_SELECCIONADA, help="Línea para headways de pares")
    parser.add_argument("--sentido", default=SENTIDO_SELECCIONADO, help="Sentido para headways de pares")
    parser.add_argument("--date", default=None, help="Filtrar por fecha YYYY-MM-DD")
    parser.add_argument("--headways-only", action="store_true", help="Solo generar headways_pares desde limpios existentes")
    return parser.parse_args()


def main():
    args = parse_args()
    eventos_dir = Path(args.eventos)
    headways_dir = Path(args.headways_dir)
    eventos_dir.mkdir(parents=True, exist_ok=True)
    headways_dir.mkdir(parents=True, exist_ok=True)

    if args.headways_only:
        limpio_files = collect_limpio_files(eventos_dir, args.date)
        if not limpio_files:
            print(f"No hay archivos *_limpio_3.csv en {eventos_dir}")
            sys.exit(1)
        print(f"--- HEADWAYS ONLY: {len(limpio_files)} archivos ---")
        process_headways(limpio_files, args.linea, args.sentido, headways_dir)
        errors = validate_limpios(limpio_files, headways_dir, args.linea, args.sentido)
    else:
        input_path = args.input or DEFAULT_RAW
        in_path = Path(input_path)
        if in_path.is_file():
            raw_files = [in_path]
        elif in_path.is_dir():
            raw_files = sorted(f for f in in_path.glob("*.csv") if "limpio" not in f.name)
            if args.date:
                raw_files = [f for f in raw_files if args.date in f.name]
        else:
            print(f"Entrada no encontrada: {input_path}")
            sys.exit(1)

        if not raw_files:
            print(f"No hay CSV brutos en {input_path}")
            sys.exit(1)

        print(f"--- POSTPROCESS: {len(raw_files)} archivos desde {input_path} ---")
        limpio_generated = run_postprocess(
            input_path=input_path,
            output_dir=str(eventos_dir),
            save_figures=False,
            date_filter=args.date,
        )

        limpio_files = [Path(p) for p in limpio_generated] if limpio_generated else [
            limpio_path_for_raw(r, eventos_dir) for r in raw_files
            if limpio_path_for_raw(r, eventos_dir).exists()
        ]

        print(f"--- HEADWAYS PARES: {len(limpio_files)} archivos ---")
        process_headways(limpio_files, args.linea, args.sentido, headways_dir)
        errors = validate_pairs(raw_files, eventos_dir, headways_dir, args.linea, args.sentido)

    if errors:
        print("\n=== ERRORES DE VALIDACIÓN ===")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print("\n=== PIPELINE COMPLETADA ===")
    print(f"Llegadas: {eventos_dir}")
    print(f"Headways: {headways_dir}")


if __name__ == "__main__":
    main()
