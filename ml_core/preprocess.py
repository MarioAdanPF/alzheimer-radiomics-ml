"""
Pasos de preprocesamiento que comparten el notebook de preparación de datos (01) y el
pipeline de inferencia (`inference.RadiomicsPipeline`), para que un paciente nuevo se
procese exactamente igual que los de entrenamiento.
"""

# Familias de PyRadiomics que dependen de la intensidad de la imagen (brillo y textura).
# Se eliminan para conservar solo la morfología (familia `shape`).
TEXTURE_FAMILIES = ['firstorder', 'glszm', 'gldm', 'glcm', 'glrlm', 'ngtdm']

def purge_texture_features(df):
    """
    Elimina las métricas de primer orden y de textura, que dependen de la intensidad
    absoluta del escáner, y conserva únicamente las características de forma (shape).
    """
    columnas_textura = [c for c in df.columns if any(fam in c for fam in TEXTURE_FAMILIES)]
    return df.drop(columns=columnas_textura)

def normalize_by_proxy_etiv(df):
    """
    Normaliza por tamaño cerebral con el "Proxy eTIV": la suma de MeshVolume, VoxelVolume y
    SurfaceArea de todas las regiones segmentadas de cada imagen (volumen y área totales
    segmentados, no el eTIV de FreeSurfer). Cada MeshVolume, VoxelVolume y SurfaceArea
    regional se divide entre el total de su familia, así que pasa a ser una proporción.
    Longitudes, diámetros y descriptores adimensionales no se modifican.
    """
    df_norm = df.copy()
    for familia in ['shape_MeshVolume', 'shape_VoxelVolume', 'shape_SurfaceArea']:
        columnas = [c for c in df_norm.columns if c.endswith(familia)]
        total = df_norm[columnas].sum(axis=1)
        df_norm[columnas] = df_norm[columnas].div(total, axis=0)
    return df_norm
