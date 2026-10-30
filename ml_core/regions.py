"""
Nombres en español de las regiones de FreeSurfer (atlas aseg + Desikan-Killiany) y de los
descriptores de forma de PyRadiomics, para tablas y figuras.
"""

# Estructuras subcorticales y generales (aseg), sin lateralidad
_SUBCORTICAL_ES = {
    'Cerebral-White-Matter': 'Sustancia blanca cerebral',
    'Lateral-Ventricle': 'Ventrículo lateral',
    'Inf-Lat-Vent': 'Asta temporal del ventrículo lateral',
    'Cerebellum-White-Matter': 'Sustancia blanca cerebelosa',
    'Cerebellum-Cortex': 'Corteza cerebelosa',
    'Thalamus-Proper*': 'Tálamo',
    'Caudate': 'Núcleo caudado',
    'Putamen': 'Putamen',
    'Pallidum': 'Globo pálido',
    'Hippocampus': 'Hipocampo',
    'Amygdala': 'Amígdala',
    'Accumbens-area': 'Núcleo accumbens',
    'VentralDC': 'Diencéfalo ventral',
    'choroid-plexus': 'Plexo coroideo',
    'vessel': 'Vaso sanguíneo',
    'undetermined': 'Región indeterminada',
}

# Estructuras de la línea media o sin lateralidad
_MIDLINE_ES = {
    '3rd-Ventricle': 'Tercer ventrículo',
    '4th-Ventricle': 'Cuarto ventrículo',
    '5th-Ventricle': 'Quinto ventrículo',
    'Brain-Stem': 'Tronco encefálico',
    'CSF': 'Líquido cefalorraquídeo',
    'WM-hypointensities': 'Hipointensidades de sustancia blanca',
    'non-WM-hypointensities': 'Hipointensidades fuera de la sustancia blanca',
    'Optic-Chiasm': 'Quiasma óptico',
    'CC_Posterior': 'Cuerpo calloso posterior',
    'CC_Mid_Posterior': 'Cuerpo calloso medio-posterior',
    'CC_Central': 'Cuerpo calloso central',
    'CC_Mid_Anterior': 'Cuerpo calloso medio-anterior',
    'CC_Anterior': 'Cuerpo calloso anterior',
}

# Parcelas corticales del atlas Desikan-Killiany
_CORTICAL_ES = {
    'bankssts': 'Banco del surco temporal superior',
    'caudalanteriorcingulate': 'Cíngulo anterior caudal',
    'caudalmiddlefrontal': 'Frontal medio caudal',
    'cuneus': 'Cuña',
    'entorhinal': 'Corteza entorrinal',
    'frontalpole': 'Polo frontal',
    'fusiform': 'Giro fusiforme',
    'inferiorparietal': 'Parietal inferior',
    'inferiortemporal': 'Temporal inferior',
    'insula': 'Ínsula',
    'isthmuscingulate': 'Istmo del cíngulo',
    'lateraloccipital': 'Occipital lateral',
    'lateralorbitofrontal': 'Orbitofrontal lateral',
    'lingual': 'Giro lingual',
    'medialorbitofrontal': 'Orbitofrontal medial',
    'middletemporal': 'Temporal medio',
    'paracentral': 'Lóbulo paracentral',
    'parahippocampal': 'Giro parahipocampal',
    'parsopercularis': 'Pars opercularis',
    'parsorbitalis': 'Pars orbitalis',
    'parstriangularis': 'Pars triangularis',
    'pericalcarine': 'Corteza pericalcarina',
    'postcentral': 'Giro poscentral',
    'posteriorcingulate': 'Cíngulo posterior',
    'precentral': 'Giro precentral',
    'precuneus': 'Precúneo',
    'rostralanteriorcingulate': 'Cíngulo anterior rostral',
    'rostralmiddlefrontal': 'Frontal medio rostral',
    'superiorfrontal': 'Frontal superior',
    'superiorparietal': 'Parietal superior',
    'superiortemporal': 'Temporal superior',
    'supramarginal': 'Giro supramarginal',
    'temporalpole': 'Polo temporal',
    'transversetemporal': 'Temporal transverso',
}

_SIDE_ES = {'Left': 'izq.', 'Right': 'der.', 'lh': 'izq.', 'rh': 'der.'}

# Descriptores de forma de PyRadiomics
DESCRIPTOR_ES = {
    'Elongation': 'Elongación',
    'Flatness': 'Aplanamiento',
    'LeastAxisLength': 'Longitud del eje menor',
    'MajorAxisLength': 'Longitud del eje mayor',
    'MinorAxisLength': 'Longitud del eje intermedio',
    'Maximum2DDiameterColumn': 'Diámetro 2D máximo (plano fila-corte)',
    'Maximum2DDiameterRow': 'Diámetro 2D máximo (plano columna-corte)',
    'Maximum2DDiameterSlice': 'Diámetro 2D máximo (plano fila-columna)',
    'Maximum3DDiameter': 'Diámetro 3D máximo',
    'MeshVolume': 'Volumen (malla)',
    'VoxelVolume': 'Volumen (vóxeles)',
    'SurfaceArea': 'Área de superficie',
    'SurfaceVolumeRatio': 'Razón superficie/volumen',
    'Sphericity': 'Esfericidad',
}

def region_name_es(freesurfer_name):
    """Nombre en español de una región a partir de su nombre en FreeSurferColorLUT."""
    if freesurfer_name in _MIDLINE_ES:
        return _MIDLINE_ES[freesurfer_name]
    if freesurfer_name.startswith('ctx-'):
        _, side, parcel = freesurfer_name.split('-', 2)
        return f"{_CORTICAL_ES.get(parcel, parcel)} {_SIDE_ES[side]}"
    side, _, structure = freesurfer_name.partition('-')
    if side in _SIDE_ES and structure in _SUBCORTICAL_ES:
        return f"{_SUBCORTICAL_ES[structure]} {_SIDE_ES[side]}"
    return freesurfer_name

def load_region_names(lut_path='./FreeSurferColorLUT.csv'):
    """Devuelve dos dicts {LabelID: nombre}: el de FreeSurfer y su traducción al español."""
    import pandas as pd
    lut = pd.read_csv(lut_path)
    names = dict(zip(lut['No.'], lut['Label Name']))
    return names, {label: region_name_es(name) for label, name in names.items()}

def split_feature(feature):
    """
    'Label_17_original_shape_Sphericity' -> (17, 'Sphericity');
    las familias que no son de forma conservan su prefijo: (17, 'glcm_Contrast').
    """
    descriptor = feature.split('original_', 1)[1]
    return int(feature.split('_')[1]), descriptor.removeprefix('shape_')

def feature_label_es(feature, names_es):
    """Etiqueta legible de una característica: región en español + descriptor de PyRadiomics."""
    label, descriptor = split_feature(feature)
    return f"{names_es.get(label, f'Región {label}')} · {descriptor}"
