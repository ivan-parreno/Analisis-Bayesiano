import asyncio
import csv
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import requests
from dotenv import load_dotenv

load_dotenv()

APP_ID   = os.getenv("TMB_APP_ID")
APP_KEY  = os.getenv("TMB_APP_KEY")
PARAMS   = {"app_id": APP_ID, "app_key": APP_KEY}

CODI_LINIA   = 208 # Línea 33
URL_PARADES  = f"https://api.tmb.cat/v1/transit/linies/bus/{CODI_LINIA}/parades"
URL_IBUS     = "https://api.tmb.cat/v1/itransit/bus/parades/{codi_parada}"

# --- CONFIGURACIÓN OPTIMIZADA ---
CONCURRENCY = 20         # Subido a 20 para hacer ~4 olas de peticiones para las 79 paradas
ITERACIONES_MAX = 420
ESPERA_MINIMA = 30       # Segundos
INTERVALO_OBJETIVO = 60  # Segundos

def get_stops_sync() -> list[dict]:
    """Obtiene la lista maestra de paradas al iniciar el programa."""
    try:
        r = requests.get(URL_PARADES, params=PARAMS, timeout=10)
        r.raise_for_status()
        data = r.json()
        features = data.get("features") or data.get("parades") or []
        stops = []
        for f in features:
            props = f.get("properties", f)
            codi = str(props.get("CODI_PARADA") or props.get("codi_parada", ""))
            if codi:
                stops.append({
                    "codi_parada": codi,
                    "nom_parada":  props.get("NOM_PARADA")  or props.get("nom_parada", ""),
                    "ordre":       int(props.get("ORDRE_PARADA") or props.get("ordre_parada", 0)),
                })
        stops.sort(key=lambda s: s["ordre"])
        return stops
    except Exception as e:
        print(f"Error obtenint llista de parades: {e}")
        return []

async def fetch_ibus_async(client: httpx.AsyncClient, sem: asyncio.Semaphore, stop: dict, captured_at: int) -> dict:
    """Obtiene los datos de una parada concreta."""
    max_retries = 3
    data = None
    
    await asyncio.sleep(random.uniform(0.05, 0.3)) 
    
    async with sem:
        for attempt in range(max_retries):
            try:
                r = await client.get(
                    URL_IBUS.format(codi_parada=stop["codi_parada"]),
                    params=PARAMS,
                    timeout=10, 
                )
                
                if r.status_code in [502, 429]:
                    await asyncio.sleep(2 ** attempt)
                    continue
                
                r.raise_for_status()
                data = r.json()
                break 
                
            except Exception:
                if attempt == max_retries - 1:
                    return {"stop": stop, "rows": [], "failed": True}
                await asyncio.sleep(1)

    # Procesamiento de datos
    rows = []
    if not data or "parades" not in data:
        return {"stop": stop, "rows": [], "failed": False}
    
    for parada in data.get("parades", []):
        for trajecte in parada.get("linies_trajectes", []):
            if not trajecte or str(trajecte.get("codi_linia")) != str(CODI_LINIA):
                continue
            
            sentit_str = "Anada" if trajecte.get("id_sentit") == 1 else "Tornada"
            for bus in trajecte.get("propers_busos", []):
                ta = bus.get("temps_arribada")
                if not ta: continue
                
                minuts = max(0.0, round((ta - captured_at) / 60000, 1))
                try:
                    hora = datetime.fromtimestamp(ta / 1000, tz=timezone.utc).astimezone()
                    rows.append({
                        "codi_parada": stop["codi_parada"],
                        "nom_parada":  stop["nom_parada"],
                        "ordre":       stop["ordre"],
                        "nom_linia":   trajecte.get("nom_linia"),
                        "sentit":      sentit_str,
                        "desti":       trajecte.get("desti_trajecte", ""),
                        "id_bus":      bus.get("id_bus"),
                        "hora":        hora.strftime("%H:%M:%S"),
                        "minuts":      minuts,
                        "temps_arribada": ta,
                    })
                except: continue
                
    return {"stop": stop, "rows": rows, "failed": False}

async def fetch_batch(client: httpx.AsyncClient, stops: list[dict], captured_at: int) -> tuple[list[dict], list[dict]]:
    """Ejecuta el lote usando un cliente ya abierto y devuelve (validos, fallidos)."""
    sem = asyncio.Semaphore(CONCURRENCY)
    valid_rows = []
    failed_stops = []

    tasks = [fetch_ibus_async(client, sem, stop, captured_at) for stop in stops]
    results = await asyncio.gather(*tasks)
    
    for res in results:
        if res["failed"]:
            failed_stops.append(res["stop"])
        else:
            valid_rows.extend(res["rows"])
            
    return valid_rows, failed_stops

def filter_logical_order(rows: list[dict], bus_tracker: dict) -> list[dict]:
    """Filtra registros asegurando coherencia, asignando IDs temporales si faltan."""
    
    for row in rows:
        if row.get("id_bus") is None:
            desti_slug = str(row.get("desti", "S/D")).replace(" ", "")
            row["id_bus"] = f"TEMP_{desti_slug}_{row.get('codi_parada')}"

    rows.sort(key=lambda x: (str(x["id_bus"]), x["ordre"]))

    valid_data = []
    for row in rows:
        b_id = str(row["id_bus"])
        current_ordre = row["ordre"]
        current_sentit = row["sentit"]

        if b_id.startswith("TEMP_"):
            valid_data.append(row)
            continue

        if b_id in bus_tracker:
            last_state = bus_tracker[b_id]
            if last_state["sentit"] != current_sentit:
                bus_tracker[b_id] = {"ordre": current_ordre, "sentit": current_sentit}
                valid_data.append(row)
                continue
            
            if current_ordre >= (last_state["ordre"] - 1):
                bus_tracker[b_id]["ordre"] = max(last_state["ordre"], current_ordre)
                valid_data.append(row)
        else:
            bus_tracker[b_id] = {"ordre": current_ordre, "sentit": current_sentit}
            valid_data.append(row)

    return valid_data

async def main_loop():
    if not APP_ID or not APP_KEY:
        print("Error: TMB_APP_ID o TMB_APP_KEY al .env")
        sys.exit(1)

    stops = get_stops_sync()
    if not stops:
        sys.exit(1)

    print(f"{len(stops)} parades trobades. Iniciant el monitoratge continu...")
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_path = f"data/arrivals_{date_str}_8_2.csv"
    Path("data").mkdir(exist_ok=True)
    
    bus_tracker = {}

    limits = httpx.Limits(max_connections=30, max_keepalive_connections=25)
    
    async with httpx.AsyncClient(limits=limits, verify=False) as client:
        for i in range(1, ITERACIONES_MAX + 1):
            ciclo_inicio = datetime.now()
            print(f"\n{ciclo_inicio.strftime('%H:%M:%S')} - Cicle {i} de {ITERACIONES_MAX}...")
            
            captured_at = int(datetime.now(timezone.utc).timestamp() * 1000)
            
            # Pasamos el cliente reutilizable a la función
            all_rows, failed_stops = await fetch_batch(client, stops, captured_at)
            
            if failed_stops:
                print(f"  Reintentant {len(failed_stops)} parades fallades...")
                retry_rows, _ = await fetch_batch(client, failed_stops, captured_at)
                all_rows.extend(retry_rows)

            filas_originales = len(all_rows)
            all_rows = filter_logical_order(all_rows, bus_tracker)
            filas_filtradas = len(all_rows)
            
            if filas_originales > filas_filtradas:
                print(f"  Filtre lògic: Descartes {filas_originales - filas_filtradas} registres incoherents.")

            exists = Path(out_path).exists()
            with open(out_path, "a", newline="", encoding="utf-8") as f:
                fieldnames = ["captured_at", "codi_parada", "nom_parada", "ordre", "nom_linia", "sentit", "desti", "id_bus", "hora", "minuts", "temps_arribada"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not exists: writer.writeheader()
                
                captured_iso = datetime.now(timezone.utc).isoformat()
                for row in all_rows:
                    writer.writerow({**row, "captured_at": captured_iso})

            duracion_ciclo = (datetime.now() - ciclo_inicio).total_seconds()
            print(f"  Fet en {duracion_ciclo:.1f}s. {filas_filtradas} registres guardats.")

            if i < ITERACIONES_MAX:
                if duracion_ciclo >= INTERVALO_OBJETIVO:
                    tiempo_espera = ESPERA_MINIMA
                else:
                    tiempo_espera = INTERVALO_OBJETIVO - duracion_ciclo
                
                print(f"  Esperant {tiempo_espera:.1f}s fins a la pròxima petició...")
                await asyncio.sleep(tiempo_espera)

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\nCaptura aturada per l'usuari.")