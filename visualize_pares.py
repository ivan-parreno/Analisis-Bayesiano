#!/usr/bin/env python3
"""Genera HTML interactivos (stringline) para visualizar headways de pares."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plot_hours import LINEA_SELECCIONADA, SENTIDO_SELECCIONADO, cargar_datos

DEFAULT_EVENTOS = "processed/eventos"
DEFAULT_HEADWAYS = "processed/headways"
DEFAULT_OUTPUT = "processed/figures"

PALETTE = [
    "#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657",
    "#79c0ff", "#56d364", "#ff7b72", "#bc8cff", "#ffb757",
]

BASE_DATE = "2000-01-01"


def minutos_a_datetime(minutos):
    h = int(minutos // 60)
    m = int(minutos % 60)
    s = int(round((minutos % 1) * 60))
    return f"{BASE_DATE} {h:02d}:{m:02d}:{s:02d}"


def paths_for_date(fecha, linea, sentido, eventos_dir, headways_dir):
    limpio = Path(eventos_dir) / f"arrivals_{fecha}_8_2_limpio_3.csv"
    headways = Path(headways_dir) / f"headways_pares3_{linea}_{sentido}_{fecha}.csv"
    return limpio, headways


def trips_from_headways(hw: pd.DataFrame) -> set[str]:
    trips = set()
    for col in ("trip_delante", "trip_detras"):
        if col in hw.columns:
            trips.update(hw[col].dropna().astype(str))
    return trips


def build_day_figure(df: pd.DataFrame, hw: pd.DataFrame, linea: str, sentido: str, fecha: str):
    """Stringline de buses en pares + headway del par seleccionable."""
    df_linea = df[(df["nom_linia"] == linea) & (df["sentit"] == sentido)].copy()
    if df_linea.empty:
        raise ValueError(f"Sin datos para {linea} {sentido} en {fecha}")

    trips = trips_from_headways(hw)
    df_pairs = df_linea[df_linea["trip_id"].isin(trips)].copy()
    if df_pairs.empty:
        raise ValueError(f"Sin viajes de pares para {fecha}")

    df_pairs["datetime"] = pd.to_datetime(df_pairs["tiempo_minutos"].apply(minutos_a_datetime))
    parada_map = df_pairs.groupby("ordre")["codi_parada"].first().to_dict()
    ordenes = sorted(parada_map.keys())
    ticktext = [f"{o} - {parada_map[o]}" for o in ordenes]

    par_ids = sorted(hw["par_id"].unique())
    trip_to_par = {}
    for par_id in par_ids:
        sub = hw[hw["par_id"] == par_id].iloc[0]
        trip_to_par[str(sub["trip_delante"])] = par_id
        trip_to_par[str(sub["trip_detras"])] = par_id

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
        subplot_titles=(
            f"Stringline · {linea} {sentido} · {fecha}",
            "Headway del par seleccionado (min)",
        ),
    )

    trace_indices_by_par: dict[str, list[int]] = {p: [] for p in par_ids}
    trace_idx = 0

    for (bus, viaje), grupo in df_pairs.groupby(["id_bus", "viaje_n"]):
        grupo = grupo.sort_values("datetime")
        trip_id = f"{bus}_{viaje}"
        par_id = trip_to_par.get(trip_id, "")
        color = PALETTE[hash(trip_id) % len(PALETTE)]
        label = f"{par_id} · Bus {bus} V{viaje}" if par_id else f"Bus {bus} V{viaje}"

        fig.add_trace(
            go.Scatter(
                x=grupo["datetime"],
                y=grupo["ordre"],
                mode="lines+markers",
                name=label,
                legendgroup=par_id or trip_id,
                line=dict(width=1.8, color=color),
                marker=dict(size=5, color=color, opacity=0.85),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Hora: %{x|%H:%M:%S}<br>"
                    "Orden: %{y}<br>"
                    "<extra></extra>"
                ),
            ),
            row=1, col=1,
        )
        if par_id:
            trace_indices_by_par[par_id].append(trace_idx)
        trace_idx += 1

    hw_trace_start = trace_idx
    for i, par_id in enumerate(par_ids):
        sub = hw[hw["par_id"] == par_id].sort_values("ordre")
        fig.add_trace(
            go.Scatter(
                x=sub["ordre"],
                y=sub["headway_pair"],
                mode="lines+markers",
                name=f"HW {par_id}",
                line=dict(width=2, color=PALETTE[i % len(PALETTE)]),
                marker=dict(size=6),
                visible=(i == 0),
                hovertemplate=(
                    f"<b>Par {par_id}</b><br>"
                    "Orden: %{x}<br>"
                    "Headway: %{y:.1f} min<br>"
                    "<extra></extra>"
                ),
            ),
            row=2, col=1,
        )
        trace_idx += 1

    n_sl = trace_idx - len(par_ids)
    n_hw = len(par_ids)

    buttons = [{
        "label": "Todos los pares",
        "method": "update",
        "args": [
            {"visible": [True] * n_sl + [False] * n_hw},
            {"title.text": f"Stringline · {linea} {sentido} · {fecha} · Todos"},
        ],
    }]
    for i, par_id in enumerate(par_ids):
        vis = [False] * trace_idx
        for idx in trace_indices_by_par[par_id]:
            vis[idx] = True
        vis[hw_trace_start + i] = True
        hw_ini = hw[hw["par_id"] == par_id]["headway_pair"].iloc[0]
        buttons.append({
            "label": f"{par_id} ({hw_ini:.0f} min)",
            "method": "update",
            "args": [
                {"visible": vis},
                {"title.text": f"Stringline · {linea} {sentido} · {fecha} · {par_id}"},
            ],
        })

    fig.update_xaxes(title_text="Hora del día", tickformat="%H:%M", row=1, col=1)
    fig.update_yaxes(
        title_text="Parada (ordre)",
        tickmode="array",
        tickvals=ordenes,
        ticktext=ticktext,
        row=1, col=1,
    )
    fig.update_xaxes(title_text="Orden de parada", row=2, col=1)
    fig.update_yaxes(title_text="Headway (min)", row=2, col=1)

    fig.update_layout(
        height=900,
        hovermode="closest",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font=dict(family="Courier New, monospace", color="#c9d1d9"),
        legend=dict(
            orientation="v",
            x=1.01,
            y=1,
            bgcolor="rgba(22,27,34,0.95)",
            bordercolor="#30363d",
            font=dict(size=9),
        ),
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 0.0,
            "y": 1.18,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "#21262d",
            "bordercolor": "#30363d",
            "font": dict(color="#c9d1d9", size=11),
        }],
        margin=dict(l=140, r=220, t=80, b=50),
    )
    return fig, par_ids


def generate_day_html(fecha, linea, sentido, eventos_dir, headways_dir, output_dir):
    limpio, headways_path = paths_for_date(fecha, linea, sentido, eventos_dir, headways_dir)
    if not limpio.exists():
        print(f"Omitido {fecha}: no existe {limpio.name}")
        return None
    if not headways_path.exists():
        print(f"Omitido {fecha}: no existe {headways_path.name}")
        return None

    df = cargar_datos(str(limpio))
    hw = pd.read_csv(headways_path)
    if hw.empty:
        print(f"Omitido {fecha}: headways vacío")
        return None

    fig, par_ids = build_day_figure(df, hw, linea, sentido, fecha)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stringline_{linea}_{sentido}_{fecha}.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    print(f"Generado: {out_path} ({len(par_ids)} pares)")
    return out_path


def discover_dates(headways_dir, linea, sentido):
    pattern = re.compile(
        rf"headways_pares3_{re.escape(linea)}_{re.escape(sentido)}_(\d{{4}}-\d{{2}}-\d{{2}})\.csv"
    )
    dates = []
    for f in Path(headways_dir).glob("headways_pares3_*.csv"):
        m = pattern.match(f.name)
        if m:
            dates.append(m.group(1))
    return sorted(dates)


def generate_index_html(dates, linea, sentido, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    options = "\n".join(
        f'    <option value="stringline_{linea}_{sentido}_{d}.html">{d}</option>'
        for d in dates
    )
    default = dates[-1] if dates else ""
    default_src = f"stringline_{linea}_{sentido}_{default}.html" if default else ""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Visualizador headways pares · {linea} {sentido}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: "Courier New", monospace;
      background: #0d1117;
      color: #c9d1d9;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    header {{
      background: #161b22;
      border-bottom: 1px solid #30363d;
      padding: 12px 20px;
      display: flex;
      align-items: center;
      gap: 16px;
      flex-shrink: 0;
    }}
    header h1 {{ font-size: 15px; font-weight: 600; }}
    label {{ font-size: 12px; color: #7d8590; }}
    select {{
      background: #21262d;
      color: #c9d1d9;
      border: 1px solid #30363d;
      padding: 6px 10px;
      font-family: inherit;
      font-size: 12px;
      border-radius: 4px;
    }}
    iframe {{
      flex: 1;
      border: none;
      width: 100%;
      background: #0d1117;
    }}
    .meta {{ font-size: 11px; color: #7d8590; margin-left: auto; }}
  </style>
</head>
<body>
  <header>
    <h1>Headways pares · Línea {linea} · {sentido}</h1>
    <label for="fecha">Fecha:</label>
    <select id="fecha" onchange="cambiarDia(this.value)">
{options}
    </select>
    <span class="meta">{len(dates)} días · stringline + headway por par</span>
  </header>
  <iframe id="vista" src="{default_src}" title="Stringline del día"></iframe>
  <script>
    function cambiarDia(fecha) {{
      document.getElementById('vista').src = fecha;
    }}
  </script>
</body>
</html>
"""
    index_path = out_dir / "visualize.html"
    index_path.write_text(html, encoding="utf-8")
    print(f"Índice: {index_path}")
    return index_path


def generate_manifest(dates, linea, sentido, headways_dir, output_dir):
    manifest = {"linea": linea, "sentido": sentido, "dates": []}
    for fecha in dates:
        hw_path = Path(headways_dir) / f"headways_pares3_{linea}_{sentido}_{fecha}.csv"
        if not hw_path.exists():
            continue
        hw = pd.read_csv(hw_path)
        pares = (
            hw.groupby("par_id")
            .agg(
                n_registros=("ordre", "count"),
                hw_inicial=("headway_pair", "first"),
                hora=("hora_delante_str", "first"),
            )
            .reset_index()
            .to_dict(orient="records")
        )
        manifest["dates"].append({"fecha": fecha, "n_pares": len(pares), "pares": pares})

    out_path = Path(output_dir) / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {out_path}")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description="Genera HTML para visualizar headways de pares.")
    p.add_argument("--eventos", default=DEFAULT_EVENTOS)
    p.add_argument("--headways-dir", default=DEFAULT_HEADWAYS)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--linea", default=LINEA_SELECCIONADA)
    p.add_argument("--sentido", default=SENTIDO_SELECCIONADO)
    p.add_argument("--date", default=None, help="Una fecha YYYY-MM-DD")
    p.add_argument("--all", action="store_true", help="Generar HTML para todos los días")
    return p.parse_args()


def main():
    args = parse_args()
    if args.date:
        dates = [args.date]
    elif args.all:
        dates = discover_dates(args.headways_dir, args.linea, args.sentido)
    else:
        dates = discover_dates(args.headways_dir, args.linea, args.sentido)
        if dates:
            dates = [dates[-1]]

    if not dates:
        print("No hay fechas con headways_pares.")
        return

    generated = 0
    for fecha in dates:
        if generate_day_html(
            fecha, args.linea, args.sentido,
            args.eventos, args.headways_dir, args.output,
        ):
            generated += 1

    if generated:
        all_dates = discover_dates(args.headways_dir, args.linea, args.sentido)
        generate_index_html(all_dates, args.linea, args.sentido, args.output)
        generate_manifest(all_dates, args.linea, args.sentido, args.headways_dir, args.output)

    print(f"\nAbre: {Path(args.output) / 'visualize.html'}")


if __name__ == "__main__":
    main()
