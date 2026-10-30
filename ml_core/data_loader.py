import pandas as pd
from pathlib import Path
import warnings

NOMBRES_COHORTE = {0: 'Ctrl', 1: '1Y', 2: '2Y'}

def cargar_cohorte(ruta_carpeta, valor_target):
    """
    Carga todos los archivos CSV de una carpeta de cohorte (uno por imagen, en formato largo:
    una fila por región con sus métricas de PyRadiomics), ignorando 'fsaverage',
    y les asigna el Target correspondiente.
    """
    ruta_carpeta = Path(ruta_carpeta)
    lista_dfs = []

    if not ruta_carpeta.exists():
        warnings.warn(f"La ruta {ruta_carpeta} no existe.")
        return pd.DataFrame()

    for archivo in ruta_carpeta.glob("*.csv"):
        if 'fsaverage' in archivo.name:
            continue
        try:
            df_temp = pd.read_csv(archivo)
            if not df_temp.empty:
                lista_dfs.append(df_temp)
        except Exception as e:
            print(f"Error cargando {archivo.name}: {e}")

    if not lista_dfs:
        return pd.DataFrame()
    df = pd.concat(lista_dfs, ignore_index=True)
    df['Target'] = valor_target
    return df
