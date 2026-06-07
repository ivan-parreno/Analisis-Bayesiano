import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# ============================================
# CONFIGURACIÓN
# ============================================
CSV_PATH = "processed/eventos/arrivals_2026-05-14_8_2_limpio_3.csv"
LINEA_SELECCIONADA = "H8"
SENTIDOS = ["Anada", "Tornada"]
MAX_HEADWAY_MINUTOS = 17
INTERVALO_HORAS = 1
SENTIDO_SELECCIONADO = "Anada"
OUTPUT_HEADWAYS_DIR = "processed/headways"
SAVE_FIGURES = False
MAX_FALTANTES_PAR = 5


def extract_date_from_limpio(path):
    """Extrae YYYY-MM-DD del nombre de un archivo *_limpio_3.csv."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", Path(path).name)
    return match.group(1) if match else None

def contar_faltantes_por_viaje(df):
    def _contar(cadena):
        if pd.isna(cadena) or str(cadena).strip() == '':
            return 0
        return len(str(cadena).split(','))
    faltantes_por_trip = df.groupby('trip_id')['ordre_faltantes'].first().apply(_contar)
    return faltantes_por_trip

def cargar_datos(ruta_csv):
    df = pd.read_csv(ruta_csv)
    required = {'hora_paso', 'nom_linia', 'sentit', 'codi_parada', 'ordre', 'ordre_faltantes',
                'id_bus', 'viaje_n'}
    for col in required:
        if col not in df.columns:
            print(f"ERROR: columna '{col}' no encontrada. Saliendo.")
            exit(1)

    def tiempo_a_minutos(t_str):
        try:
            if pd.isna(t_str): return np.nan
            parts = str(t_str).split(':')
            if len(parts) == 3:
                h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
                return h*60 + m + s/60.0
            else:
                return float(t_str)*60
        except:
            return np.nan

    df['tiempo_minutos'] = df['hora_paso'].apply(tiempo_a_minutos)
    df = df.dropna(subset=['tiempo_minutos'])
    df['trip_id'] = df['id_bus'].astype(str) + '_' + df['viaje_n'].astype(str)
    # num_faltantes por viaje+sentido (trip_id solo no basta: Anada y Tornada comparten id)
    def _contar_faltantes(cadena):
        if pd.isna(cadena):
            return 0
        s = str(cadena).strip()
        if s == '' or s.lower() == 'nan':
            return 0
        return len(s.split(','))

    df['_trip_sentit'] = df['trip_id'] + '_' + df['sentit']
    df_ordenado = df[df['ordre'] >= 1].sort_values(['_trip_sentit', 'ordre'])
    faltantes_series = (
        df_ordenado.groupby('_trip_sentit')['ordre_faltantes']
        .first()
        .apply(_contar_faltantes)
    )
    df['num_faltantes'] = df['_trip_sentit'].map(faltantes_series).fillna(0).astype(int)
    df.drop(columns=['_trip_sentit'], inplace=True)
    df['valido'] = df['num_faltantes'] == 0
    total = len(df)
    validos = df['valido'].sum()
    print(f"Registros totales: {total}, viajes completos (sin ordre_faltantes): {validos} ({validos/total*100:.1f}%)")
    return df

# ============================================
# 2. HEADWAY POR PARADA (SIN FILTRO DE VALIDEZ)
# ============================================
def calcular_headways_parada(df_parada, max_headway_min=None):
    df_parada = df_parada.sort_values('tiempo_minutos').copy()
    if len(df_parada) < 2:
        return pd.DataFrame(columns=['hora', 'headway_min'])

    df_parada['tiempo_ant'] = df_parada['tiempo_minutos'].shift(1)
    df_parada['faltante_ant'] = df_parada['ordre_faltantes'].shift(1)
    df_parada['headway_min'] = df_parada['tiempo_minutos'] - df_parada['tiempo_ant']

    es_faltante_actual = df_parada['ordre_faltantes'].notna() & (df_parada['ordre_faltantes'].astype(str).str.strip() != '')
    es_faltante_anterior = df_parada['faltante_ant'].notna() & (df_parada['faltante_ant'].astype(str).str.strip() != '')
    mascara_limpios = (~es_faltante_actual) & (~es_faltante_anterior)
    df_hw = df_parada[mascara_limpios].copy()
    df_hw = df_hw.rename(columns={'tiempo_minutos': 'hora'})
    df_hw['hora'] = df_hw['hora'] / 60.0
    df_hw = df_hw[df_hw['headway_min'] > 0]
    if max_headway_min is not None:
        df_hw = df_hw[df_hw['headway_min'] <= max_headway_min]
    return df_hw[['hora', 'headway_min']]

def estadisticas_por_parada(df, linea, sentido=None, max_headway_seg=1020):
    mask = (df['nom_linia'] == linea)
    if sentido:
        mask &= (df['sentit'] == sentido)
    df_filt = df[mask].copy()
    if df_filt.empty:
        return pd.DataFrame()

    max_headway_min = max_headway_seg / 60.0 if max_headway_seg else None
    resultados = []
    for cod_parada, grupo in df_filt.groupby('codi_parada'):
        hw_df = calcular_headways_parada(grupo, max_headway_min=max_headway_min)
        if hw_df.empty:
            continue
        ordre_val = grupo['ordre'].iloc[0]
        media = hw_df['headway_min'].mean()
        sigma = hw_df['headway_min'].std()
        resultados.append({
            'ordre': ordre_val,
            'codi_parada': cod_parada,
            'headway_medio_min': media,
            'headway_sigma_min': sigma,
            'count': len(hw_df)
        })
    df_res = pd.DataFrame(resultados)
    if not df_res.empty:
        df_res = df_res.sort_values('ordre')
    return df_res

def analizar_tendencia_horaria(df, linea, sentido=None, intervalo_horas=1, max_headway_seg=1020):
    mask = (df['nom_linia'] == linea)
    if sentido:
        mask &= (df['sentit'] == sentido)
    df_filt = df[mask].copy()
    if df_filt.empty:
        return pd.DataFrame()

    max_headway_min = max_headway_seg / 60.0 if max_headway_seg else None
    todos_hw = []
    for cod_parada, grupo in df_filt.groupby('codi_parada'):
        hw_df = calcular_headways_parada(grupo, max_headway_min=max_headway_min)
        todos_hw.append(hw_df)
    if not todos_hw:
        return pd.DataFrame()
    df_hw = pd.concat(todos_hw, ignore_index=True)
    df_hw['hora_redondeada'] = (df_hw['hora'] // intervalo_horas) * intervalo_horas
    agrupado = df_hw.groupby('hora_redondeada')['headway_min'].agg(['mean', 'std', 'count']).reset_index()
    return agrupado

def graficar_headway_diario(df, linea, sentido=None, cod_parada=None, max_headway_min=MAX_HEADWAY_MINUTOS):
    mask = (df['nom_linia'] == linea)
    if sentido:
        mask &= (df['sentit'] == sentido)
    if cod_parada:
        mask &= (df['codi_parada'] == cod_parada)
    df_filt = df[mask].copy()
    if df_filt.empty:
        print("No hay datos para la selección indicada.")
        return

    frames = []
    for parada, grupo in df_filt.groupby('codi_parada'):
        hw_parada = calcular_headways_parada(grupo, max_headway_min=max_headway_min)
        hw_parada['codi_parada'] = parada
        frames.append(hw_parada)
    if not frames:
        print("No se generaron headways.")
        return
    hw_todos = pd.concat(frames, ignore_index=True)

    plt.figure(figsize=(14, 6))
    if cod_parada:
        plt.scatter(hw_todos['hora'], hw_todos['headway_min'], alpha=0.6, s=30, color='teal')
        plt.title(f'Headway diario - Línea {linea} ({sentido}) - Parada {cod_parada}')
    else:
        colores_paradas = {p: f'C{i}' for i, p in enumerate(hw_todos['codi_parada'].unique())}
        for p, grp in hw_todos.groupby('codi_parada'):
            plt.scatter(grp['hora'], grp['headway_min'], alpha=0.5, s=20, label=f'Parada {p}', color=colores_paradas[p])
        plt.legend(markerscale=2, fontsize='small', ncol=2)
        plt.title(f'Headway diario - Línea {linea} ({sentido}) - Todas las paradas')
    plt.xlabel('Hora del día')
    plt.ylabel('Headway (minutos)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    # plt.show()

# ============================================
# 3. ACF
# ============================================
def plot_acf_todas_las_paradas(df, linea, sentido, save_figures=SAVE_FIGURES):
    mask = (df['nom_linia'] == linea) & (df['sentit'] == sentido) & df['valido']
    df_filt = df[mask].copy()
    if df_filt.empty:
        print(f"No hay viajes completos para Línea {linea}, sentido {sentido}.")
        return

    paradas_info = []
    for cod_parada, grupo in df_filt.groupby('codi_parada'):
        hw_df = calcular_headways_parada(grupo, max_headway_min=None)
        if len(hw_df) < 6:
            continue
        headways = hw_df['headway_min'].values
        paradas_info.append({
            'ordre': grupo['ordre'].iloc[0],
            'codi_parada': cod_parada,
            'headways': headways
        })

    if not paradas_info:
        print("No hay suficientes observaciones para ACF.")
        return

    paradas_info = sorted(paradas_info, key=lambda x: x['ordre'])
    num_paradas = len(paradas_info)
    cols = 4
    rows = int(np.ceil(num_paradas / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3.2 * rows), sharey=True)
    axes = axes.flatten()

    for idx, info in enumerate(paradas_info):
        ax = axes[idx]
        hw = info['headways']
        max_lags = min(10, len(hw) // 2)
        plot_acf(hw, lags=max_lags, ax=ax, alpha=0.05)
        ax.set_title(f"Ord {info['ordre']} - Parada {info['codi_parada']}", fontsize=10, fontweight='bold')
        ax.set_xlabel('Lags', fontsize=8)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=8)

    for i in range(num_paradas, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(f"ACF por Parada (solo viajes completos) - Línea {linea} [{sentido}]", y=0.99, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_figures:
        plt.savefig(f"acf_grid_{linea}_{sentido}.png", dpi=300, bbox_inches='tight')
    plt.close()

# ============================================
# 4. ANÁLISIS DE PARES
# ============================================
def preparar_datos_pares(df, linea, sentido):
    mask = (df['nom_linia'] == linea) & (df['sentit'] == sentido) & df['valido']
    df_filt = df[mask][['trip_id', 'codi_parada', 'ordre', 'tiempo_minutos']].copy()
    return df_filt.sort_values(['trip_id', 'ordre'])

def emparejar_viajes(df_trips):
    primera_parada = df_trips['ordre'].min()
    df_primera = df_trips[df_trips['ordre'] == primera_parada][['trip_id', 'tiempo_minutos']]
    df_primera = df_primera.sort_values('tiempo_minutos')
    trips_ordenados = df_primera['trip_id'].tolist()
    pares = [(trips_ordenados[i], trips_ordenados[i+1]) for i in range(len(trips_ordenados)-1)]
    return {par: idx for idx, par in enumerate(pares)}

def calcular_headways_pares(df_trips, pares_dict):
    pares_inv = {v: k for k, v in pares_dict.items()}
    frames = []
    for par_id, (trip_ant, trip_act) in pares_inv.items():
        df_ant = df_trips[df_trips['trip_id'] == trip_ant][['codi_parada', 'ordre', 'tiempo_minutos']]
        df_act = df_trips[df_trips['trip_id'] == trip_act][['codi_parada', 'ordre', 'tiempo_minutos']]
        merged = pd.merge(df_ant, df_act, on=['codi_parada', 'ordre'], suffixes=('_ant', '_act'))
        if merged.empty:
            continue
        merged['headway_pair'] = merged['tiempo_minutos_act'] - merged['tiempo_minutos_ant']
        merged = merged[merged['headway_pair'] > 0]
        merged['par_id'] = par_id
        frames.append(merged[['par_id', 'ordre', 'codi_parada', 'headway_pair']])
    if not frames:
        return pd.DataFrame(columns=['par_id', 'ordre', 'codi_parada', 'headway_pair'])
    return pd.concat(frames, ignore_index=True).sort_values(['par_id', 'ordre'])

def variabilidad_por_par(hw_pares):
    variab = hw_pares.groupby('par_id')['headway_pair'].std().reset_index()
    variab.columns = ['par_id', 'sigma_headway']
    return variab

def puntos_criticos(hw_pares):
    df = hw_pares.sort_values(['par_id', 'ordre'])
    df['delta_headway'] = df.groupby('par_id')['headway_pair'].diff()
    df_delta = df.dropna(subset=['delta_headway'])
    criticidad = df_delta.groupby(['ordre', 'codi_parada'])['delta_headway'].apply(
        lambda x: x.abs().mean()
    ).reset_index(name='cambio_medio_absoluto')
    return criticidad.sort_values('cambio_medio_absoluto', ascending=False)

def plot_headway_pares(hw_pares, max_pares=30):
    pares_muestra = hw_pares['par_id'].unique()[:max_pares]
    df_plot = hw_pares[hw_pares['par_id'].isin(pares_muestra)]
    plt.figure(figsize=(12, 6))
    for pid, grp in df_plot.groupby('par_id'):
        plt.plot(grp['ordre'], grp['headway_pair'], alpha=0.4, linewidth=1)
    plt.xlabel('Orden de parada')
    plt.ylabel('Headway del par (min)')
    plt.title('Evolución del headway entre viajes completos consecutivos')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    # plt.show()

def plot_puntos_criticos(df_criticidad, top_n=10):
    top = df_criticidad.nlargest(top_n, 'cambio_medio_absoluto')
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top)), top['cambio_medio_absoluto'], tick_label=top['codi_parada'])
    plt.xlabel('Cambio absoluto medio del headway (min)')
    plt.title('Paradas con mayor variación del headway entre pares')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    # plt.show()

def detectar_pares_persistentes(df_trips, max_headway_inicial=20,
                                min_paradas_comunes=6,
                                umbral_persistencia=0.9,
                                hora_min=None, hora_max=None):
    viajes = df_trips['trip_id'].unique()
    n = len(viajes)
    print(f"Evaluando {n*(n-1)//2} posibles pares de viajes...")

    frames_pares = []
    for i in range(n):
        trip_a = viajes[i]
        df_a = df_trips[df_trips['trip_id'] == trip_a]
        for j in range(i+1, n):
            trip_b = viajes[j]
            df_b = df_trips[df_trips['trip_id'] == trip_b]
            merged = pd.merge(df_a, df_b, on=['codi_parada', 'ordre'], suffixes=('_a', '_b'))
            if len(merged) < min_paradas_comunes:
                continue
            merged['dif'] = merged['tiempo_minutos_b'] - merged['tiempo_minutos_a']
            adelante_a = (merged['dif'] > 0).sum()
            adelante_b = (merged['dif'] < 0).sum()
            total = len(merged)
            if adelante_a >= total * umbral_persistencia:
                predecesor = 'a'
                merged['headway_pair'] = merged['dif']
            elif adelante_b >= total * umbral_persistencia:
                predecesor = 'b'
                merged['headway_pair'] = -merged['dif']
            else:
                continue
            idx_min = merged['ordre'].idxmin()
            hw_inicial = merged.loc[idx_min, 'headway_pair']
            hora_inicio = (merged.loc[idx_min, 'tiempo_minutos_b'] if predecesor == 'a'
                           else merged.loc[idx_min, 'tiempo_minutos_a']) / 60.0
            if hw_inicial > max_headway_inicial:
                continue
            if hora_min is not None and hora_inicio < hora_min:
                continue
            if hora_max is not None and hora_inicio > hora_max:
                continue
            if predecesor == 'a':
                par_id = f"{trip_a}__{trip_b}"
            else:
                par_id = f"{trip_b}__{trip_a}"
            merged['par_id'] = par_id
            frames_pares.append(merged[['par_id', 'ordre', 'codi_parada', 'headway_pair']])

    if not frames_pares:
        return pd.DataFrame(columns=['par_id', 'ordre', 'codi_parada', 'headway_pair'])

    hw_pares = pd.concat(frames_pares, ignore_index=True)
    return hw_pares.sort_values(['par_id', 'ordre'])

def limpiar_pares_duplicados(hw_pares):
    if hw_pares.empty:
        return hw_pares

    headway_inicial = hw_pares.groupby('par_id').apply(
        lambda g: g.loc[g['ordre'].idxmin(), 'headway_pair']
    ).reset_index(name='hw_inicial')
    hw = hw_pares.merge(headway_inicial, on='par_id', how='left')
    if 'trip_delante' not in hw.columns or 'trip_detras' not in hw.columns:
        hw[['trip_delante', 'trip_detras']] = hw['par_id'].str.split('__', expand=True)

    cambio = True
    while cambio:
        cambio = False
        duplicados_lider = hw.groupby('trip_delante')['par_id'].nunique()
        lideres_con_conflicto = duplicados_lider[duplicados_lider > 1].index
        for trip_lider in lideres_con_conflicto:
            subset = hw[hw['trip_delante'] == trip_lider]
            mejor_par_id = subset.loc[subset['hw_inicial'].idxmin(), 'par_id']
            pares_a_eliminar = subset[subset['par_id'] != mejor_par_id]['par_id'].unique()
            if len(pares_a_eliminar) > 0:
                hw = hw[~hw['par_id'].isin(pares_a_eliminar)]
                cambio = True
        duplicados_seguidor = hw.groupby('trip_detras')['par_id'].nunique()
        seguidores_con_conflicto = duplicados_seguidor[duplicados_seguidor > 1].index
        for trip_seguidor in seguidores_con_conflicto:
            subset = hw[hw['trip_detras'] == trip_seguidor]
            mejor_par_id = subset.loc[subset['hw_inicial'].idxmin(), 'par_id']
            pares_a_eliminar = subset[subset['par_id'] != mejor_par_id]['par_id'].unique()
            if len(pares_a_eliminar) > 0:
                hw = hw[~hw['par_id'].isin(pares_a_eliminar)]
                cambio = True
    return hw.drop(columns=['hw_inicial'])

def preparar_datos_pares_flexible(df, linea, sentido, max_faltantes=5):
    mask = (
        (df['nom_linia'] == linea)
        & (df['sentit'] == sentido)
        & (df['ordre'] >= 1)
        & (df['num_faltantes'] <= max_faltantes)
    )
    df_filt = df[mask][['trip_id', 'codi_parada', 'ordre', 'tiempo_minutos']].copy()
    return df_filt.sort_values(['trip_id', 'ordre'])

def plot_headway_pares_con_tendencia(hw_pares, df_criticidad=None, top_n_criticas=5, save_figures=SAVE_FIGURES, linea=LINEA_SELECCIONADA, sentido=SENTIDO_SELECCIONADO):
    if hw_pares.empty:
        print("No hay datos de pares para graficar.")
        return

    tendencia = hw_pares.groupby('ordre')['headway_pair'].quantile([0.25, 0.5, 0.75]).unstack()
    tendencia.columns = ['q25', 'median', 'q75']
    tendencia = tendencia.reset_index()

    plt.figure(figsize=(14, 6))
    for par_id, grp in hw_pares.groupby('par_id'):
        plt.plot(grp['ordre'], grp['headway_pair'],
                 alpha=0.15, linewidth=0.7, color='steelblue')
    plt.fill_between(tendencia['ordre'], tendencia['q25'], tendencia['q75'],
                     color='navy', alpha=0.2, label='IQR (Q1‑Q3)')
    plt.plot(tendencia['ordre'], tendencia['median'],
             color='navy', linewidth=2.5, label='Mediana')

    if df_criticidad is not None and not df_criticidad.empty:
        top = df_criticidad.nlargest(top_n_criticas, 'cambio_medio_absoluto')
        for _, row in top.iterrows():
            orden_crit = row['ordre']
            codigo_crit = row['codi_parada']
            if orden_crit in tendencia['ordre'].values:
                mediana_val = tendencia.loc[tendencia['ordre'] == orden_crit, 'median'].values[0]
                plt.scatter(orden_crit, mediana_val, color='red', s=60, zorder=5,
                            edgecolors='darkred')
                plt.annotate(codigo_crit, (orden_crit, mediana_val),
                             textcoords="offset points", xytext=(0,12),
                             ha='center', fontsize=8, color='red', weight='bold')

    plt.xlabel('Orden de parada')
    plt.ylabel('Headway del par (minutos)')
    plt.title('Evolución del headway a lo largo de la línea (todos los pares válidos)')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    if save_figures:
        plt.savefig(f"headway_pares_tendencia_{linea}_{sentido}.png", dpi=300, bbox_inches='tight')
    plt.close()

def anadir_horas_pares(df, hw_pares):
    if hw_pares.empty:
        return hw_pares
    if 'trip_delante' not in hw_pares.columns or 'trip_detras' not in hw_pares.columns:
        hw_pares[['trip_delante', 'trip_detras']] = hw_pares['par_id'].str.split('__', expand=True)

    primera_parada = hw_pares.groupby('par_id')['ordre'].min().reset_index(name='primera_ordre')
    hw_primera = pd.merge(hw_pares, primera_parada, on='par_id')
    hw_primera = hw_primera[hw_primera['ordre'] == hw_primera['primera_ordre']].copy()
    hw_primera = hw_primera[['par_id', 'trip_delante', 'trip_detras', 'codi_parada', 'ordre']].drop_duplicates()
    df_horas = df[['trip_id', 'codi_parada', 'ordre', 'tiempo_minutos']].copy()

    horas_delante = pd.merge(hw_primera[['par_id', 'trip_delante', 'codi_parada', 'ordre']],
                             df_horas,
                             left_on=['trip_delante', 'codi_parada', 'ordre'],
                             right_on=['trip_id', 'codi_parada', 'ordre'],
                             how='left')
    horas_delante = horas_delante[['par_id', 'tiempo_minutos']].rename(columns={'tiempo_minutos': 'hora_delante'})

    horas_detras = pd.merge(hw_primera[['par_id', 'trip_detras', 'codi_parada', 'ordre']],
                            df_horas,
                            left_on=['trip_detras', 'codi_parada', 'ordre'],
                            right_on=['trip_id', 'codi_parada', 'ordre'],
                            how='left')
    horas_detras = horas_detras[['par_id', 'tiempo_minutos']].rename(columns={'tiempo_minutos': 'hora_detras'})

    hw_pares = hw_pares.merge(horas_delante, on='par_id', how='left')
    hw_pares = hw_pares.merge(horas_detras, on='par_id', how='left')
    hw_pares['franja_hora'] = (hw_pares['hora_delante'] // 60).astype(int)

    def minutos_a_hhmmss(minutos):
        if pd.isna(minutos):
            return ''
        h = int(minutos // 60)
        m = int(minutos % 60)
        s = int(round((minutos % 1) * 60))
        return f"{h:02d}:{m:02d}:{s:02d}"

    hw_pares['hora_delante_str'] = hw_pares['hora_delante'].apply(minutos_a_hhmmss)
    hw_pares['hora_detras_str']  = hw_pares['hora_detras'].apply(minutos_a_hhmmss)
    return hw_pares

# ============================================
# NUEVA FUNCIÓN: Stringline interactivo
# ============================================
def plot_stringline(df, linea, sentidos=None, output_html=None, save_figures=SAVE_FIGURES):
    if sentidos is None:
        sentidos = df['sentit'].unique().tolist()
    elif isinstance(sentidos, str):
        sentidos = [sentidos]

    df_linea = df[(df['nom_linia'] == linea) & (df['sentit'].isin(sentidos))].copy()
    if df_linea.empty:
        print("No hay datos para el stringline.")
        return

    BASE_DATE = "2000-01-01"
    def minutos_a_datetime(minutos):
        h = int(minutos // 60)
        m = int(minutos % 60)
        s = int(round((minutos % 1) * 60))
        return f"{BASE_DATE} {h:02d}:{m:02d}:{s:02d}"

    df_linea['datetime'] = pd.to_datetime(df_linea['tiempo_minutos'].apply(minutos_a_datetime))
    parada_map = df_linea.groupby('ordre')['codi_parada'].first().to_dict()
    ordenes_unicos = sorted(parada_map.keys())
    tickvals = ordenes_unicos
    ticktext = [f"{o} - {parada_map[o]}" for o in ordenes_unicos]

    PALETTE = [
        '#58a6ff','#3fb950','#f78166','#d2a8ff','#ffa657',
        '#79c0ff','#56d364','#ff7b72','#bc8cff','#ffb757',
        '#a5d6ff','#85e89d','#ffab70','#b392f0','#ffd33d',
    ]

    groups = df_linea.groupby(['id_bus', 'viaje_n'])
    bus_ids = df_linea['id_bus'].unique()
    bus_color = {bus: PALETTE[i % len(PALETTE)] for i, bus in enumerate(bus_ids)}

    fig = go.Figure()
    for (bus, viaje), grupo in groups:
        grupo = grupo.sort_values('datetime')
        label = f"Bus {bus} · V{viaje}"
        color = bus_color[bus]
        fig.add_trace(go.Scatter(
            x=grupo['datetime'],
            y=grupo['ordre'],
            mode='lines+markers',
            name=label,
            # Sin legendgroup → cada viaje es independiente en la leyenda
            line=dict(width=1.8, color=color),
            marker=dict(size=5, color=color, opacity=0.85),
            connectgaps=False,
            hovertemplate=(
                f"<b>Bus {bus} · Viaje {viaje}</b><br>"
                "Hora: %{x|%H:%M:%S}<br>"
                "Orden parada: %{y}<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text=f"Diagrama Espacio-Tiempo · Línea {linea} ({', '.join(sentidos)})",
            font=dict(size=14, color='#c9d1d9', family='Courier New, monospace'),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(text='Hora del día', font=dict(size=11, color='#7d8590')),
            type='date',
            tickformat='%H:%M',
            gridcolor='#21262d',
            gridwidth=1,
            zeroline=False,
            color='#7d8590',
            tickfont=dict(family='Courier New, monospace', size=10, color='#7d8590'),
        ),
        yaxis=dict(
            title=dict(text='Parada (ordre)', font=dict(size=11, color='#7d8590')),
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext,
            gridcolor='#21262d',
            gridwidth=1,
            zeroline=False,
            color='#7d8590',
            tickfont=dict(family='Courier New, monospace', size=9, color='#7d8590'),
        ),
        legend=dict(
            title=dict(text='Bus · Viaje (clic para ocultar)', font=dict(size=10, color='#7d8590')),
            orientation='v',
            x=1.01,
            y=1,
            xanchor='left',
            bgcolor='rgba(22,27,34,0.95)',
            bordercolor='#30363d',
            borderwidth=1,
            font=dict(family='Courier New, monospace', size=9, color='#c9d1d9'),
            # itemsizing='constant'  # opcional, para que los iconos no varíen
        ),
        hovermode='closest',
        margin=dict(l=130, r=200, t=50, b=50),  # un poco más ancho para la leyenda
        plot_bgcolor='#0d1117',
        paper_bgcolor='#161b22',
        font=dict(family='Courier New, monospace', color='#c9d1d9'),
        height=720,
    )

    if save_figures and output_html:
        fig.write_html(output_html, include_plotlyjs='cdn')
        print(f"Stringline guardado en {output_html}")
    elif save_figures:
        fig.show()

# ============================================
# NUEVA FUNCIÓN: Pirámide de headway medio y sigma
# ============================================
def plot_headway_pyramid(comparativa, titulo="Headway: media y variabilidad por parada",
                         archivo_salida=None, save_figures=SAVE_FIGURES):
    """
    Gráfico de barras divergentes (pirámide) que muestra:
      - hacia la izquierda: headway medio
      - hacia la derecha: desviación típica (sigma)
    para cada parada. Permite comparar las estadísticas 'todos' y 'solo pares'.
    """
    if comparativa.empty:
        print("Comparativa vacía, no se genera pirámide.")
        return

    # Ordenar por ordre
    df = comparativa.sort_values('ordre').copy()
    stops = df['codi_parada'].astype(str) + " (ord " + df['ordre'].astype(str) + ")"
    y_pos = range(len(stops))

    fig, ax = plt.subplots(figsize=(10, max(6, len(stops)*0.4)))

    # Barras de media (hacia la izquierda) y sigma (hacia la derecha)
    # Todos los buses
    ax.barh(y_pos, -df['headway_medio_min_todos'], height=0.6,
            color='steelblue', alpha=0.9, label='Media (todos)')
    ax.barh(y_pos, df['headway_sigma_min_todos'], height=0.6,
            color='darkorange', alpha=0.9, label='Sigma (todos)')

    # Si existen las columnas de "solo pares", las añadimos como barras más finas
    if 'headway_medio_min_solo_pares' in df.columns:
        ax.barh(y_pos, -df['headway_medio_min_solo_pares'], height=0.3,
                color='navy', alpha=0.7, label='Media (solo pares)')
    if 'headway_sigma_min_solo_pares' in df.columns:
        ax.barh(y_pos, df['headway_sigma_min_solo_pares'], height=0.3,
                color='firebrick', alpha=0.7, label='Sigma (solo pares)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stops, fontsize=8)
    ax.axvline(0, color='white', linewidth=1)
    ax.set_xlabel('Minutos')
    ax.set_title(titulo, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    # Ajustar límites simétricos
    max_val = max(df['headway_medio_min_todos'].max(), df['headway_sigma_min_todos'].max()) * 1.1
    ax.set_xlim(-max_val, max_val)

    plt.tight_layout()
    if save_figures and archivo_salida:
        plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
        print(f"Pirámide guardada en {archivo_salida}")
    plt.close()


def export_headways_pares(
    csv_path,
    linea=LINEA_SELECCIONADA,
    sentido=SENTIDO_SELECCIONADO,
    output_dir=OUTPUT_HEADWAYS_DIR,
    max_headway_minutos=MAX_HEADWAY_MINUTOS,
    max_faltantes=MAX_FALTANTES_PAR,
    save_figures=SAVE_FIGURES,
):
    """Genera el CSV de headways entre pares para un archivo limpio."""
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fecha = extract_date_from_limpio(csv_path)
    if not fecha:
        raise ValueError(f"No se pudo extraer fecha de: {csv_path}")

    df = cargar_datos(str(csv_path))
    df_pares_flex = preparar_datos_pares_flexible(df, linea, sentido, max_faltantes=max_faltantes)

    if df_pares_flex.empty:
        print(f"Sin viajes para {linea} {sentido} en {csv_path.name}")
        return None

    hw_pares = detectar_pares_persistentes(
        df_pares_flex,
        max_headway_inicial=max_headway_minutos,
        min_paradas_comunes=6,
        umbral_persistencia=0.9,
    )
    if hw_pares.empty:
        print(f"Sin pares persistentes en {csv_path.name}")
        return None

    hw_pares = limpiar_pares_duplicados(hw_pares)
    hw_pares = anadir_horas_pares(df, hw_pares)

    df_exportar = hw_pares.copy()
    if 'trip_delante' not in df_exportar.columns:
        df_exportar[['trip_delante', 'trip_detras']] = df_exportar['par_id'].str.split('__', expand=True)
    df_exportar['bus_delante'] = df_exportar['trip_delante'].apply(lambda x: "_".join(x.split('_')[:-1]))
    df_exportar['bus_detras'] = df_exportar['trip_detras'].apply(lambda x: "_".join(x.split('_')[:-1]))

    columnas_csv = [
        'par_id', 'bus_delante', 'trip_delante', 'bus_detras', 'trip_detras',
        'codi_parada', 'ordre', 'headway_pair',
        'hora_delante_str', 'hora_detras_str', 'franja_hora',
    ]
    columnas_existentes = [c for c in columnas_csv if c in df_exportar.columns]
    df_exportar = df_exportar[columnas_existentes]

    out_path = output_dir / f"headways_pares3_{linea}_{sentido}_{fecha}.csv"
    df_exportar.to_csv(out_path, index=False, sep=',')
    print(f"CSV exportado: {out_path} ({len(df_exportar)} registros)")
    return out_path


# ============================================
# MAIN
# ============================================
def run_full_analysis(
    csv_path=CSV_PATH,
    linea=LINEA_SELECCIONADA,
    sentido=SENTIDO_SELECCIONADO,
    output_dir=OUTPUT_HEADWAYS_DIR,
    save_figures=SAVE_FIGURES,
):
    df = cargar_datos(csv_path)

    # --- HEADWAY POR PARADA ---
    print("\n" + "="*60)
    print("HEADWAY POR PARADA (todos los buses, sin filtrar por ordre_faltantes)")
    print("="*60)
    for sentido in SENTIDOS:
        df_stats = estadisticas_por_parada(df, LINEA_SELECCIONADA, sentido,
                                           max_headway_seg=MAX_HEADWAY_MINUTOS*60)
        if not df_stats.empty:
            print(f"=== SENTIDO {sentido} ===")
            print(f"{'Ordre':<6} {'Parada':<10} {'Headway medio (min)':<20} {'Sigma (min)':<15} {'N obs':<10}")
            for _, row in df_stats.iterrows():
                print(f"{row['ordre']:<6} {row['codi_parada']:<10} {row['headway_medio_min']:<20.1f} {row['headway_sigma_min']:<15.1f} {row['count']:<10}")
            print()

    # --- ACF ---
    print("\n" + "="*60)
    print(f"ANÁLISIS DE AUTOCORRELACIÓN (ACF) - SOLO VIAJES COMPLETOS - SENTIDO {SENTIDO_SELECCIONADO}")
    print("="*60)
    plot_acf_todas_las_paradas(df, LINEA_SELECCIONADA, SENTIDO_SELECCIONADO, save_figures=save_figures)

    # --- HEADWAY DIARIO ---
    print("\n" + "="*60)
    print("GRÁFICO DE HEADWAY DIARIO (todos los buses, sentido Anada)")
    print("="*60)
    graficar_headway_diario(df, LINEA_SELECCIONADA, sentido="Anada", max_headway_min=MAX_HEADWAY_MINUTOS)

    # --- ANÁLISIS DE PARES ---
    print("\n" + "="*60)
    print(f"ANÁLISIS DE HEADWAY ENTRE VIAJES CON ≤{MAX_FALTANTES_PAR} FALTANTES (PARES PERSISTENTES)")
    print("="*60)

    df_pares_flex = preparar_datos_pares_flexible(df, LINEA_SELECCIONADA, SENTIDO_SELECCIONADO, max_faltantes=MAX_FALTANTES_PAR)
    total_viajes = df_pares_flex['trip_id'].nunique()
    print(f"Viajes considerados (≤{MAX_FALTANTES_PAR} faltantes): {total_viajes}")

    if df_pares_flex.empty:
        print(f"No hay viajes para Línea {LINEA_SELECCIONADA} sentido {SENTIDO_SELECCIONADO} con ≤{MAX_FALTANTES_PAR} faltantes.")
        hw_pares = pd.DataFrame()
    else:
        hw_pares = detectar_pares_persistentes(
            df_pares_flex,
            max_headway_inicial=MAX_HEADWAY_MINUTOS,
            min_paradas_comunes=6,
            umbral_persistencia=0.9
        )
        if not hw_pares.empty:
            print(f"Pares antes de limpieza: {hw_pares['par_id'].nunique()}")
            hw_pares = limpiar_pares_duplicados(hw_pares)
            print(f"Pares después de limpieza: {hw_pares['par_id'].nunique()}")
            hw_pares = anadir_horas_pares(df, hw_pares)
            print("Horas añadidas a los pares (hora_delante_str, hora_detras_str, franja_hora)")

    # Calcular trips_en_pares (reutilizable)
    trips_en_pares = set()
    if not hw_pares.empty:
        for par_id in hw_pares['par_id'].unique():
            t_a, t_b = par_id.split('__')
            trips_en_pares.add(t_a)
            trips_en_pares.add(t_b)
        print(f"Viajes únicos en pares: {len(trips_en_pares)}")

    # --- 5. COMPARATIVA ---
    comparativa = pd.DataFrame()
    if not hw_pares.empty:
        df_tradicional = estadisticas_por_parada(df, LINEA_SELECCIONADA, SENTIDO_SELECCIONADO,
                                                 max_headway_seg=MAX_HEADWAY_MINUTOS*60)
        df_pares_mean = hw_pares.groupby(['ordre', 'codi_parada'])['headway_pair'].mean().reset_index()
        df_pares_mean.columns = ['ordre', 'codi_parada', 'headway_pares_mean']
        df_pares_std = hw_pares.groupby(['ordre', 'codi_parada'])['headway_pair'].std().reset_index()
        df_pares_std.columns = ['ordre', 'codi_parada', 'headway_pares_std']

        comparativa = pd.merge(df_tradicional, df_pares_mean, on=['ordre', 'codi_parada'], how='inner')
        comparativa = pd.merge(comparativa, df_pares_std, on=['ordre', 'codi_parada'], how='inner')
        comparativa.rename(columns={'headway_medio_min': 'headway_medio_min_todos',
                                    'headway_sigma_min': 'headway_sigma_min_todos'}, inplace=True)

        # Headway solo pares (tradicional pero filtrado)
        df_solo_pares = df[df['trip_id'].isin(trips_en_pares)].copy()
        df_tradicional_pares = estadisticas_por_parada(df_solo_pares, LINEA_SELECCIONADA, SENTIDO_SELECCIONADO,
                                                       max_headway_seg=MAX_HEADWAY_MINUTOS*60)
        if not df_tradicional_pares.empty:
            comparativa = pd.merge(comparativa,
                                   df_tradicional_pares[['ordre', 'codi_parada', 'headway_medio_min', 'headway_sigma_min']],
                                   on=['ordre', 'codi_parada'],
                                   suffixes=('', '_solo_pares'))
            comparativa.rename(columns={'headway_medio_min': 'headway_medio_min_solo_pares',
                                        'headway_sigma_min': 'headway_sigma_min_solo_pares'}, inplace=True)

    # --- EXPORTACIÓN CSV ---
    export_headways_pares(
        csv_path,
        linea=LINEA_SELECCIONADA,
        sentido=SENTIDO_SELECCIONADO,
        output_dir=output_dir,
        save_figures=save_figures,
    )

    # --- STRINGLINE GENERAL ---
    print("\n" + "="*60)
    print("DIAGRAMA ESPACIO-TIEMPO (STRINGLINE) - TODOS LOS BUSES")
    print("="*60)
    plot_stringline(
        df, LINEA_SELECCIONADA, sentidos="Anada",
        output_html=f"stringline_{LINEA_SELECCIONADA}_Anada.html",
        save_figures=save_figures,
    )

    # --- STRINGLINE SOLO PARES ---
    if not hw_pares.empty and trips_en_pares:
        print("\n" + "="*60)
        print("DIAGRAMA ESPACIO-TIEMPO (STRINGLINE) - SOLO BUSES DE LOS PARES")
        print("="*60)
        df_pares_only = df[df['trip_id'].isin(trips_en_pares)].copy()
        if not df_pares_only.empty:
            plot_stringline(
                df_pares_only, LINEA_SELECCIONADA, sentidos="Anada",
                output_html=f"stringline_{LINEA_SELECCIONADA}_Anada_solo_pares.html",
                save_figures=save_figures,
            )
        else:
            print("No hay datos para stringline solo pares.")
    else:
        print("\nNo hay pares para generar stringline solo pares.")

    # --- PIRÁMIDE DE HEADWAY ---
    if not comparativa.empty:
        print("\n" + "="*60)
        print("PIRÁMIDE HEADWAY MEDIO vs SIGMA")
        print("="*60)
        plot_headway_pyramid(
            comparativa,
            titulo=f"Línea {LINEA_SELECCIONADA} ({SENTIDO_SELECCIONADO}) - Media y Sigma por parada",
            archivo_salida=f"pyramid_headway_{LINEA_SELECCIONADA}_{SENTIDO_SELECCIONADO}.png",
            save_figures=save_figures,
        )
    else:
        print("\nNo se generó comparativa para la pirámide.")


def parse_args():
    parser = argparse.ArgumentParser(description="Análisis de headways y exportación de pares.")
    parser.add_argument("--csv", default=CSV_PATH, help="Archivo *_limpio_3.csv de entrada")
    parser.add_argument("--linea", default=LINEA_SELECCIONADA, help="Línea a analizar")
    parser.add_argument("--sentido", default=SENTIDO_SELECCIONADO, help="Sentido (Anada/Tornada)")
    parser.add_argument("--output-dir", default=OUTPUT_HEADWAYS_DIR, help="Carpeta de salida para headways_pares")
    parser.add_argument("--export-only", action="store_true", help="Solo exportar CSV de pares, sin análisis completo")
    parser.add_argument("--figures", action="store_true", help="Generar gráficos (PNG/HTML)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.export_only:
        export_headways_pares(
            args.csv,
            linea=args.linea,
            sentido=args.sentido,
            output_dir=args.output_dir,
            save_figures=args.figures,
        )
    else:
        run_full_analysis(
            csv_path=args.csv,
            linea=args.linea,
            sentido=args.sentido,
            output_dir=args.output_dir,
            save_figures=args.figures,
        )