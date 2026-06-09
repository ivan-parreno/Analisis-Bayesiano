"""Configuración compartida del pipeline bayesiano."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRACES_DIR = ROOT / "traces"
OUTPUTS_DIR = ROOT / "outputs"

EVENTOS_DIR = ROOT.parent / "processed" / "eventos"
HEADWAYS_DIR = ROOT.parent / "processed" / "headways"

LINEA = "H8"
SENTIDO = "Anada"

Y_SHIFT = 0.16          # min — soporte Gamma desplazada
HW_MIN = 1.0            # min
HW_MAX = 30.0           # min
N_FRANJAS = 17          # 0=madrugada, 1-15=07-21h, 16=noche
N_WEEKDAYS = 7


def hora_a_franja(hora_reloj: int) -> int:
    """Mapea hora 0-23 al código de franja del plan."""
    if hora_reloj < 7:
        return 0
    if hora_reloj >= 22:
        return 16
    return hora_reloj - 6   # 7→1, 8→2, …, 21→15


def tiempo_str_a_minutos(t_str) -> float:
    import pandas as pd
    if pd.isna(t_str):
        return float("nan")
    parts = str(t_str).split(":")
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 60 + m + s / 60.0
    return float(t_str) * 60
