"""
Gráficas de la tesis. Solo presentación: los cálculos se hacen en los notebooks y en
`evaluation.py`; aquí se dibujan y se guardan en `img/`.

Criterio: tipos de gráfica sencillos y conocidos (barras, nubes de puntos, dona, matrices,
líneas, histogramas) y el menor texto posible dentro de la imagen: ejes, categorías, valores
y una leyenda corta. La explicación de cada figura va en el notebook y en el pie de figura.

Todas las funciones guardan la figura (PNG a 300 dpi) y devuelven su ruta; con `show=False`
se guarda sin mostrarse. Las que aceptan `ax` pueden dibujarse como panel de una figura mayor:
en ese caso no guardan nada, y quien creó la figura la guarda con `save_figure`.
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.preprocessing import RobustScaler, StandardScaler

from .regions import feature_label_es

# --- Estilo común -------------------------------------------------------------------------
INK, INK_2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
HIGHLIGHT, NEUTRAL = '#2a78d6', '#b0aea6'
POSITIVE, NEGATIVE = '#e34948', '#2a78d6'
# Un color fijo por cohorte (paleta validada para daltonismo); siempre el mismo en todas las figuras
COHORT_COLORS = {'Ctrl': '#2a78d6', '1Y': '#eb6834', '2Y': '#1baf7a'}
COHORT_LEGEND = {'Ctrl': 'Ctrl: DCL estable', '1Y': '1Y: 12 meses antes del diagnóstico',
                 '2Y': '2Y: 24 meses antes del diagnóstico'}
# Cada comparativa toma el color de su cohorte conversora; 1Y vs 2Y, que no tiene modelo, va en gris
COMPARATIVA_COLORS = {'1Y_vs_Ctrl': COHORT_COLORS['1Y'], '2Y_vs_Ctrl': COHORT_COLORS['2Y'], '1Y_vs_2Y': MUTED}
METRIC_LABELS = {'AUC': 'AUC', 'Exactitud_Balanceada': 'Exactitud\nbalanceada', 'Sensibilidad': 'Sensibilidad',
                 'Especificidad': 'Especificidad', 'F1_Macro': 'F1-macro'}
MODEL_LABELS = {'Radiómica (ajustada)': 'Radiómica', 'Radiómica (sin ajuste)': 'Radiómica\nsin ajuste',
                'Todos los VoxelVolume (ajustado)': 'Volumen de todas\nlas regiones',
                'Hipocampo (ajustado)': 'Volumen del\nhipocampo', 'Solo edad + sexo': 'Edad y sexo',
                'Regresion_Logistica': 'Regresión\nlogística', 'Random_Forest': 'Random\nForest'}
# Color y trazo de cada modelo cuando varios comparten una gráfica de líneas
MODEL_STYLES = {'Radiómica (ajustada)': (HIGHLIGHT, '-'), 'Hipocampo (ajustado)': (INK_2, '--'),
                'Solo edad + sexo': (MUTED, ':')}
STEP_LABELS = {'entrada': 'Entrada', 'ajuste': 'Ajuste por edad y sexo', 'varianza': 'Filtro de varianza',
               'escalado': 'Escalado robusto', 'correlacion': 'Filtro de correlación', 'seleccion': 'Selección Lasso'}
FAMILY_LABELS = {'shape': 'Forma', 'firstorder': 'Primer orden', 'glcm': 'Textura GLCM', 'glrlm': 'Textura GLRLM',
                 'glszm': 'Textura GLSZM', 'gldm': 'Textura GLDM', 'ngtdm': 'Textura NGTDM'}

mpl.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK, 'axes.labelcolor': INK_2,
    'axes.edgecolor': AXIS, 'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10.5, 'legend.frameon': False,
    'figure.dpi': 110,
})

def comparativa_label(comparativa):
    """'1Y_vs_Ctrl' -> '1Y vs Ctrl'."""
    return comparativa.replace('_vs_', ' vs ')

def _comparativa_color(name, default=HIGHLIGHT):
    return COMPARATIVA_COLORS.get(name.replace(' vs ', '_vs_'), default)

def _model_label(name, one_line=False):
    label = MODEL_LABELS.get(name, name.replace('_', ' '))
    return label.replace('\n', ' ') if one_line else label

def _signed(value, decimals):
    """Número con signo explícito y el signo menos tipográfico."""
    return f"{value:+.{decimals}f}".replace('-', '−')

def save_figure(filename, output_dir='./img/', show=True):
    """Ajusta los márgenes de la figura activa, la guarda en `output_dir` y la muestra o la cierra."""
    plt.gcf().tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return path

def _own_axes(ax, figsize):
    """Devuelve (ax, True) creando una figura nueva si no se pasó `ax`."""
    if ax is not None:
        return ax, False
    _, ax = plt.subplots(figsize=figsize)
    return ax, True

def _finish(own, filename, output_dir, show):
    return save_figure(filename, output_dir, show) if own else None

def _value_label(ax, x, y, text, horizontal=False, **kwargs):
    """Valor junto al extremo de una barra."""
    options = dict(fontsize=10, ha='left' if horizontal else 'center', va='center' if horizontal else 'bottom')
    options.update(kwargs)
    ax.annotate(text, (x, y), xytext=(4, 0) if horizontal else (0, 3), textcoords='offset points', **options)

def _grouped_bars(ax, groups, series, values, colors, errors=None, fmt='{:.3f}', hatches=None, fontsize=9.5):
    """
    Barras verticales agrupadas: `values[serie][grupo]`. Devuelve la posición x de cada barra
    como {(serie, grupo): x}. Los valores ausentes no se dibujan.
    """
    width = 0.8 / len(series)
    positions = {}
    for j, serie in enumerate(series):
        for i, group in enumerate(groups):
            value = values[serie].get(group)
            if value is None:
                continue
            x = i + (j - (len(series) - 1) / 2) * width
            positions[(serie, group)] = x
            error = errors[serie][group] if errors else 0
            ax.bar(x, value, width * 0.92, color=colors[serie], yerr=error or None, ecolor=INK, capsize=2.5,
                   error_kw=dict(elinewidth=1), hatch=hatches.get((serie, group)) if hatches else None,
                   edgecolor='white', linewidth=0)
            if value >= 0:
                _value_label(ax, x, value + error, fmt.format(value).replace('-', '−'), fontsize=fontsize)
            else:   # valores negativos: la etiqueta va debajo del extremo de la barra
                ax.annotate(fmt.format(value).replace('-', '−'), (x, value - error), xytext=(0, -3), textcoords='offset points',
                            ha='center', va='top', fontsize=fontsize)
    ax.set_xticks(range(len(groups)), groups)
    ax.grid(axis='x', visible=False)
    return positions

def _bars_by_cohort(ax, values, ylabel, errors=None, fmt='{:.0f}'):
    """Una barra por cohorte, con su valor encima."""
    for i, cohorte in enumerate(COHORT_COLORS):
        error = errors[cohorte] if errors else 0
        ax.bar(i, values[cohorte], width=0.6, color=COHORT_COLORS[cohorte], yerr=error or None, ecolor=INK, capsize=4,
               error_kw=dict(elinewidth=1.2))
        _value_label(ax, i, values[cohorte] + error, fmt.format(values[cohorte]), fontsize=12, fontweight='bold')
    ax.set_xticks(range(3), list(COHORT_COLORS), fontsize=12.5)
    ax.set_ylabel(ylabel)
    ax.grid(axis='x', visible=False)

# --- Notebook 01: preparación de datos ----------------------------------------------------

def plot_cohort_distribution(imagenes, filename='Distribucion_Cohortes', output_dir='./img/', show=True):
    """Dona con el número y el porcentaje de imágenes de cada cohorte."""
    _, ax = plt.subplots(figsize=(8.4, 5.2))
    values = [imagenes[c] for c in COHORT_COLORS]
    total = sum(values)
    wedges, _ = ax.pie(values, colors=list(COHORT_COLORS.values()), startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor='white', linewidth=2.5))
    for wedge, value in zip(wedges, values):
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        ax.text(0.79 * np.cos(angle), 0.79 * np.sin(angle), f"{value}\n({value / total:.0%})", ha='center', va='center',
                fontsize=13, fontweight='bold', color='white')
    ax.text(0, 0, f"N = {total}", ha='center', va='center', fontsize=18, fontweight='bold')
    ax.legend(wedges, [COHORT_LEGEND[c] for c in COHORT_COLORS], loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=12)
    return save_figure(filename, output_dir, show)

def plot_region_prevalence(prevalencia, names_es, threshold, filename='QC_Prevalencia_Regiones',
                           output_dir='./img/', show=True):
    """Barras: regiones que no están en todas las imágenes, las exigidas y las opcionales."""
    data = prevalencia[prevalencia < 1].sort_values()
    _, ax = plt.subplots(figsize=(8.6, 0.44 * len(data) + 1.2))
    positions = np.arange(len(data))
    ax.barh(positions, 100 * data.values, color=[HIGHLIGHT if p >= threshold else NEUTRAL for p in data], height=0.66)
    for pos, p in zip(positions, data.values):
        _value_label(ax, 100 * p, pos, f"{100 * p:.1f} %", horizontal=True)
    ax.set_yticks(positions, [names_es.get(l, str(l)) for l in data.index])
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Imágenes con la región (%)')
    ax.grid(axis='y', visible=False)
    ax.legend(handles=[Patch(color=HIGHLIGHT, label='Exigida'), Patch(color=NEUTRAL, label='Opcional')], loc='lower right')
    return save_figure(filename, output_dir, show)

def plot_invalid_volumes(df_largo, patient_id, labels, names_es, column='original_shape_MeshVolume',
                         filename='QC_Volumenes_No_Positivos', output_dir='./img/', show=True):
    """
    Barras pareadas por región: el valor de la imagen excluida y la mediana de las demás imágenes.
    La figura no muestra el identificador de la imagen (dato de ADNI).
    """
    resto = df_largo[df_largo['PatientID'] != patient_id]
    imagen = df_largo[df_largo['PatientID'] == patient_id].set_index('LabelID')[column]
    medianas = resto[resto['LabelID'].isin(labels)].groupby('LabelID')[column].median()
    _, ax = plt.subplots(figsize=(8.6, 0.52 * len(labels) + 1.4))
    positions = np.arange(len(labels))
    ax.barh(positions + 0.2, [medianas[l] for l in labels], height=0.38, color=NEUTRAL, label='Mediana de las demás imágenes')
    ax.barh(positions - 0.2, [imagen[l] for l in labels], height=0.38, color=POSITIVE, label='Imagen excluida')
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(positions, [names_es.get(l, str(l)) for l in labels])
    ax.set_xlabel(f"{column.split('shape_')[1]} (mm³)")
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower right')
    return save_figure(filename, output_dir, show)

def plot_feature_families(descriptores_por_familia, filename='Familias_Radiomicas', output_dir='./img/', show=True):
    """Barras: descriptores de PyRadiomics por familia; solo la familia de forma se conserva."""
    data = descriptores_por_familia.sort_values()
    _, ax = plt.subplots(figsize=(8, 0.5 * len(data) + 1.2))
    positions = np.arange(len(data))
    ax.barh(positions, data.values, color=[HIGHLIGHT if f == 'shape' else NEUTRAL for f in data.index], height=0.66)
    for pos, n in zip(positions, data.values):
        _value_label(ax, n, pos, f"{int(n)}", horizontal=True, fontsize=11)
    ax.set_yticks(positions, [FAMILY_LABELS.get(f, f) for f in data.index])
    ax.set_xlim(0, data.max() * 1.1)
    ax.set_xlabel('Descriptores por región')
    ax.grid(axis='y', visible=False)
    ax.legend(handles=[Patch(color=HIGHLIGHT, label='Se conserva'), Patch(color=NEUTRAL, label='Se elimina')], loc='lower right')
    return save_figure(filename, output_dir, show)

def plot_total_volume(total_litros, cohortes, filename='Volumen_Total_Segmentado', output_dir='./img/', show=True):
    """Barras: media (± DE) del volumen total segmentado por cohorte."""
    values = np.asarray(total_litros, dtype=float)
    _, ax = plt.subplots(figsize=(5.6, 4.6))
    _bars_by_cohort(ax, {c: values[cohortes == c].mean() for c in COHORT_COLORS}, 'Volumen total segmentado (L)',
                    errors={c: values[cohortes == c].std() for c in COHORT_COLORS}, fmt='{:.2f}')
    return save_figure(filename, output_dir, show)

# Nubes de puntos con el estilo de la figura de PCA de la tesis (antes y después de ComBat):
# un marcador por cohorte, borde blanco, cuadrícula completa y leyenda con marco
COHORT_MARKERS = {'Ctrl': 'o', '1Y': 'X', '2Y': 's'}

def _cohort_scatter(ax, x, y, cohortes):
    for cohorte in COHORT_COLORS:
        mask = cohortes == cohorte
        ax.scatter(x[mask], y[mask], s=62, marker=COHORT_MARKERS[cohorte], color=COHORT_COLORS[cohorte], alpha=0.85,
                   edgecolors='white', linewidths=0.6, label=cohorte)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.9)
    ax.legend(title='Cohorte', frameon=True, framealpha=0.95, edgecolor=AXIS, markerscale=1.1, fontsize=11, title_fontsize=11.5)

def plot_volume_vs_total(total_litros, values, cohortes, ylabel, filename, output_dir='./img/', show=True):
    """Nube de puntos: volumen de una región frente al volumen total segmentado, por cohorte."""
    total_litros, values = np.asarray(total_litros, dtype=float), np.asarray(values, dtype=float)
    _, ax = plt.subplots(figsize=(7.4, 5.6))
    _cohort_scatter(ax, total_litros, values, cohortes)
    ax.set_xlabel('Volumen total segmentado (L)')
    ax.set_ylabel(ylabel)
    return save_figure(filename, output_dir, show)

def plot_pca_distribution(X_features, y_target, names_es=None, scaler='robust', zoom=True, filename='PCA_Morfologico',
                          output_dir='./img/', show=True, ax=None, title=None):
    """
    Nube de puntos de las imágenes en las dos primeras componentes principales, por cohorte.
    `scaler`: 'robust' (RobustScaler) o 'standard' (StandardScaler). Con `zoom`, los ejes ignoran
    el 1 % de valores extremos. Imprime las características con más peso en cada componente.
    """
    pca = PCA(n_components=2, random_state=42)
    X_scaled = (StandardScaler() if scaler == 'standard' else RobustScaler()).fit_transform(X_features)
    X_pca = pca.fit_transform(X_scaled)
    for i, component in enumerate(['PC1', 'PC2']):
        print(f"{component} explica el {pca.explained_variance_ratio_[i]:.1%} de la varianza. Características con más peso:")
        loadings = pd.Series(pca.components_[i], index=X_features.columns)
        for feature in loadings.abs().sort_values(ascending=False).head(5).index:
            print(f"   {loadings[feature]:+.3f}  {feature_label_es(feature, names_es or {})}")

    cohortes = pd.Series(np.asarray(y_target)).map({0: 'Ctrl', 1: '1Y', 2: '2Y'}).to_numpy()
    ax, own = _own_axes(ax, (8, 6.2))
    _cohort_scatter(ax, X_pca[:, 0], X_pca[:, 1], cohortes)
    if title:
        ax.set_title(title)
    if zoom:
        for values, set_limits in [(X_pca[:, 0], ax.set_xlim), (X_pca[:, 1], ax.set_ylim)]:
            low, high = np.percentile(values, [1, 99])
            set_limits(low - 0.35 * (high - low), high + 0.35 * (high - low))
    ax.set_xlabel(f"Componente Principal 1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"Componente Principal 2 ({pca.explained_variance_ratio_[1]:.0%})")
    return _finish(own, filename, output_dir, show)

def plot_pca_comparison(X_todas, X_forma, y_target, names_es=None, filename='PCA_Comparacion', output_dir='./img/', show=True):
    """
    Dos PCA lado a lado: todas las familias de PyRadiomics sin normalizar (StandardScaler, como el
    análisis de ComBat) y solo las características de forma normalizadas (las que usan los modelos).
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    print("--- Todas las características, sin normalizar ---")
    plot_pca_distribution(X_todas, y_target, names_es, scaler='standard', zoom=False, ax=axes[0],
                          title='Todas las características')
    print("--- Solo características de forma, normalizadas ---")
    plot_pca_distribution(X_forma, y_target, names_es, scaler='robust', zoom=True, ax=axes[1],
                          title='Solo características de forma')
    return save_figure(filename, output_dir, show)

def plot_demographics(demografia, filename='Demografia_Cohortes', output_dir='./img/', show=True):
    """Barras: edad media (± DE), proporción de mujeres y proporción de imágenes del periodo ADNI1, por cohorte."""
    cohortes = demografia['Target'].map({0: 'Ctrl', 1: '1Y', 2: '2Y'}).to_numpy()
    _, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    edad = demografia['Edad'].to_numpy(dtype=float)
    _bars_by_cohort(axes[0], {c: edad[cohortes == c].mean() for c in COHORT_COLORS}, 'Edad (años)',
                    errors={c: edad[cohortes == c].std() for c in COHORT_COLORS}, fmt='{:.1f}')
    _bars_by_cohort(axes[1], {c: 100 * demografia.loc[cohortes == c, 'Sexo_F'].mean() for c in COHORT_COLORS},
                    'Mujeres (%)', fmt='{:.0f} %')
    _bars_by_cohort(axes[2], {c: 100 * demografia.loc[cohortes == c, 'Periodo_ADNI1'].mean() for c in COHORT_COLORS},
                    'Imágenes de la fase ADNI1 (%)', fmt='{:.0f} %')
    for ax in axes[1:]:
        ax.set_ylim(0, 100)
    return save_figure(filename, output_dir, show)

# --- Notebook 02: modelo y métricas -------------------------------------------------------

def plot_pipeline_diagram(filename='Diagrama_Pipeline', output_dir='./img/', show=True):
    """Diagrama de flujo de los pasos del pipeline de clasificación (en azul, los que se ajustan con el entrenamiento)."""
    steps = [('Características de forma', False), ('Ajuste por edad y sexo', True), ('Filtro de varianza', True),
             ('Escalado robusto', True), ('Filtro de correlación', True), ('Selección Lasso', True),
             ('Regresión logística', True), ('Probabilidad de conversión', False)]
    _, ax = plt.subplots(figsize=(4.6, 7.6))
    height, gap, width = 0.62, 0.34, 4.2
    for i, (name, fitted) in enumerate(steps):
        y = -(i * (height + gap))
        ax.add_patch(FancyBboxPatch((0, y - height), width, height, boxstyle='round,pad=0.02,rounding_size=0.12',
                                    facecolor='#eaf2fc' if fitted else '#f3f2ee', edgecolor=HIGHLIGHT if fitted else AXIS, linewidth=1.5))
        ax.text(width / 2, y - height / 2, name, fontsize=13, fontweight='bold', va='center', ha='center')
        if i < len(steps) - 1:
            ax.annotate('', xy=(width / 2, y - height - gap + 0.03), xytext=(width / 2, y - height - 0.03),
                        arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.5))
    ax.set_xlim(-0.1, width + 0.1)
    ax.set_ylim(-(len(steps) * (height + gap)) + gap - 0.1, 0.1)
    ax.axis('off')
    return save_figure(filename, output_dir, show)

def plot_nested_cv_diagram(n_folds=5, filename='Diagrama_CV_Anidada', output_dir='./img/', show=True):
    """Esquema de la validación cruzada anidada: la prueba rota entre los folds y el entrenamiento se vuelve a dividir."""
    train_color, test_color, val_color = '#cfe1f7', '#eb6834', '#1baf7a'
    _, ax = plt.subplots(figsize=(8.4, 6.2))
    x0, block = 1.0, 1.5
    width = n_folds * block

    def grid(top, row_height, row_gap, hold_color, fontsize):
        """Dibuja n_folds filas; en la fila i el bloque i es el que se aparta."""
        for i in range(n_folds):
            y = top - i * (row_height + row_gap)
            ax.text(x0 - 0.15, y + row_height / 2, f"Fold {i + 1}", ha='right', va='center', fontsize=fontsize, color=INK_2)
            for j in range(n_folds):
                ax.add_patch(Rectangle((x0 + j * block, y), block - 0.05, row_height,
                                       facecolor=hold_color if i == j else train_color, edgecolor='none'))
        return top - (n_folds - 1) * (row_height + row_gap)

    outer_bottom = grid(6.0, 0.44, 0.12, test_color, 11)
    # El entrenamiento del último fold se amplía en la validación interna
    train_right = x0 + (n_folds - 1) * block - 0.05
    ax.add_patch(Rectangle((x0, outer_bottom), train_right - x0, 0.44, facecolor='none', edgecolor=INK, linewidth=1.8))
    inner_top = outer_bottom - 1.0
    ax.plot([x0, x0], [outer_bottom, inner_top + 0.32], color=INK_2, lw=1, ls=':')
    ax.plot([train_right, x0 + width - 0.05], [outer_bottom, inner_top + 0.32], color=INK_2, lw=1, ls=':')
    inner_bottom = grid(inner_top, 0.32, 0.09, val_color, 10)
    ax.legend(handles=[Patch(color=train_color, label='Entrenamiento'), Patch(color=test_color, label='Prueba'),
                       Patch(color=val_color, label='Validación interna')], loc='upper center',
              bbox_to_anchor=(0.55, 0.06), ncol=3, fontsize=11.5)
    ax.set_xlim(0, x0 + width + 0.1)
    ax.set_ylim(inner_bottom - 0.75, 6.6)
    ax.axis('off')
    return save_figure(filename, output_dir, show)

def plot_fold_anatomy(columns_by_step, thresholds, f1_scores, threshold, filename='Anatomia_Fold',
                      output_dir='./img/', show=True):
    """Características que deja cada paso del pipeline en un fold (barras) y F1-macro según el umbral (línea)."""
    _, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    names = [STEP_LABELS.get(name, name) for name, _ in columns_by_step][::-1]
    counts = [n for _, n in columns_by_step][::-1]
    positions = np.arange(len(names))
    ax.barh(positions, counts, color=[HIGHLIGHT] + [NEUTRAL] * (len(names) - 1), height=0.66)
    for pos, n in zip(positions, counts):
        _value_label(ax, n, pos, f"{n}", horizontal=True, fontsize=11.5, fontweight='bold')
    ax.set_xlim(0, max(counts) * 1.14)
    ax.set_yticks(positions, names)
    ax.set_xlabel('Características')
    ax.grid(axis='y', visible=False)

    ax = axes[1]
    f1_scores = np.asarray(f1_scores)
    best = float(f1_scores[np.argmin(np.abs(np.asarray(thresholds) - threshold))])
    ax.plot(thresholds, f1_scores, color=HIGHLIGHT, lw=2.2)
    ax.axvline(threshold, color=INK, ls='--', lw=1.2)
    ax.scatter([threshold], [best], s=90, color=INK, zorder=4, edgecolor='white', linewidth=1.4)
    ax.annotate(f"{threshold:.2f}", (threshold, best), xytext=(7, 7), textcoords='offset points', fontsize=12, fontweight='bold')
    ax.set_xlabel('Umbral de decisión')
    ax.set_ylabel('F1-macro')
    return save_figure(filename, output_dir, show)

def plot_auc_by_fold(results_by_comparativa, filename='AUC_por_Fold', output_dir='./img/', show=True):
    """Nube de puntos: AUC de cada partición de prueba por comparativa, con su media."""
    _, ax = plt.subplots(figsize=(7.6, 4.8))
    rng = np.random.default_rng(0)
    for i, (comparativa, results) in enumerate(results_by_comparativa.items()):
        aucs = results['folds']['AUC'].to_numpy()
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(aucs)), aucs, s=46, color=_comparativa_color(comparativa), alpha=0.6, linewidths=0)
        ax.plot([i - 0.28, i + 0.28], [aucs.mean()] * 2, color=INK, lw=2.4, zorder=4)
        ax.annotate(f"{aucs.mean():.3f}", (i + 0.3, aucs.mean()), xytext=(4, 0), textcoords='offset points', va='center',
                    fontsize=12, fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', pad=1.5), zorder=5)
    ax.axhline(0.5, color=INK_2, ls='--', lw=1, zorder=1)
    ax.set_xticks(range(len(results_by_comparativa)), [comparativa_label(c) for c in results_by_comparativa], fontsize=12.5)
    ax.set_xlim(-0.5, len(results_by_comparativa) - 0.25)
    ax.set_ylabel('AUC')
    ax.grid(axis='x', visible=False)
    ax.legend(handles=[Line2D([], [], color=INK, lw=2.4, label='Media'), Line2D([], [], color=INK_2, ls='--', lw=1, label='Azar')],
              loc='lower left')
    return save_figure(filename, output_dir, show)

def plot_metrics_panel(results_by_comparativa, filename='Metricas_Modelo', output_dir='./img/', show=True):
    """Barras agrupadas: media (± DE) de cada métrica en las particiones de prueba, por comparativa."""
    metrics = list(METRIC_LABELS)
    labels = [METRIC_LABELS[m] for m in metrics]
    series = list(results_by_comparativa)
    _, ax = plt.subplots(figsize=(11.6, 4.8))
    means = {c: dict(zip(labels, r['folds'][metrics].mean())) for c, r in results_by_comparativa.items()}
    sds = {c: dict(zip(labels, r['folds'][metrics].std())) for c, r in results_by_comparativa.items()}
    _grouped_bars(ax, labels, series, means, COMPARATIVA_COLORS, errors=sds)
    ax.axhline(0.5, color=INK_2, ls='--', lw=1)
    ax.set_ylim(0, 1.05)
    ax.legend(handles=[Patch(color=COMPARATIVA_COLORS[c], label=comparativa_label(c)) for c in series]
              + [Line2D([], [], color=INK_2, ls='--', lw=1, label='Azar')], loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=4)
    return save_figure(filename, output_dir, show)

def plot_mean_roc(results_by_model, title=None, filename='ROC', output_dir='./img/', show=True, ax=None):
    """Curva ROC media sobre las particiones de prueba, para uno o varios modelos ({nombre: resultados})."""
    ax, own = _own_axes(ax, (5.6, 5.4))
    fpr_grid = np.linspace(0, 1, 101)
    fallback = iter(['#eda100', '#e87ba4', '#2a78d6', '#eb6834', '#1baf7a'])
    for name, results in results_by_model.items():
        color, linestyle = MODEL_STYLES.get(name, (COMPARATIVA_COLORS.get(name.replace(' vs ', '_vs_')), '-'))
        color = color or next(fallback)
        y = results['y']
        curves = []
        for r, _, _, idx_test in results['splits']:
            fpr, tpr, _ = roc_curve(y[idx_test], results['probs_oof'][r, idx_test])
            curves.append(np.interp(fpr_grid, fpr, tpr))
            curves[-1][0] = 0.0
        ax.plot(fpr_grid, np.mean(curves, axis=0), lw=2.4, color=color, ls=linestyle,
                label=f"{_model_label(name, one_line=True)} (AUC {results['folds']['AUC'].mean():.3f})")
    ax.plot([0, 1], [0, 1], color=INK_2, ls='--', lw=1, label='Azar')
    ax.set_xlabel('1 − Especificidad')
    ax.set_ylabel('Sensibilidad')
    ax.set_aspect('equal')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='lower right', fontsize=10)
    if title:
        ax.set_title(title)
    return _finish(own, filename, output_dir, show)

def plot_oof_confusion(results, class_names, comparativa, filename=None, output_dir='./img/', show=True, ax=None):
    """
    Matriz de confusión agregada: cada imagen se clasifica cuando está en la partición de prueba, con el
    umbral de su fold. Conteo medio por repetición y porcentaje por clase real.
    """
    ax, own = _own_axes(ax, (4.8, 4.6))
    y = results['y']
    cm = np.zeros((2, 2))
    for (r, _, _, idx_test), threshold in zip(results['splits'], results['folds']['Umbral']):
        cm += confusion_matrix(y[idx_test], (results['probs_oof'][r, idx_test] >= threshold).astype(int), labels=[0, 1])
    cm_mean = cm / results['probs_oof'].shape[0]
    cm_pct = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_mean[i, j]:.0f}\n({cm_pct[i, j]:.0%})", ha='center', va='center', fontsize=15,
                    fontweight='bold', color='white' if cm_pct[i, j] > 0.55 else INK)
    ax.set_xticks([0, 1], class_names, fontsize=12.5)
    ax.set_yticks([0, 1], class_names, fontsize=12.5)
    ax.set_xlabel('Clase predicha')
    ax.set_ylabel('Clase real')
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(comparativa_label(comparativa))
    return _finish(own, filename or f"CM_{comparativa}", output_dir, show)

def plot_coefficients(stability, names_es, comparativa, top=15, filename=None, output_dir='./img/', show=True):
    """
    Dos gráficas de barras de las variables seleccionadas con más frecuencia en el bootstrap:
    su coeficiente medio (izquierda) y su frecuencia de selección (derecha).
    """
    data = stability.head(top).iloc[::-1]
    positions = np.arange(len(data))
    _, (ax, ax_freq) = plt.subplots(1, 2, figsize=(11, 0.4 * len(data) + 1.6), sharey=True,
                                    gridspec_kw={'width_ratios': [2, 1]})
    ax.barh(positions, data['Coef_Medio'], color=[POSITIVE if c > 0 else NEGATIVE for c in data['Coef_Medio']], height=0.68)
    ax.axvline(0, color=INK, lw=1)
    limit = data['Coef_Medio'].abs().max() * 1.12
    ax.set_xlim(-limit, limit)
    ax.set_yticks(positions, [feature_label_es(c, names_es) for c in data.index])
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Coeficiente')
    ax.grid(axis='y', visible=False)
    ax.set_title(comparativa_label(comparativa))

    ax_freq.barh(positions, 100 * data['Frecuencia_Seleccion'], color=NEUTRAL, height=0.68)
    for pos, f in zip(positions, data['Frecuencia_Seleccion']):
        _value_label(ax_freq, 100 * f, pos, f"{f:.0%}", horizontal=True)
    ax_freq.set_xlim(0, 118)
    ax_freq.set_xticks([0, 50, 100])
    ax_freq.set_xlabel('Frecuencia (%)')
    ax_freq.tick_params(axis='y', length=0)
    ax_freq.grid(axis='y', visible=False)
    return save_figure(filename or f"Coef_{comparativa}", output_dir, show)

def plot_feature_elimination(curve, k_best, k_1se, comparativa, k_max_display=None, filename=None,
                             output_dir='./img/', show=True, ax=None):
    """
    Gráfica de líneas: AUC medio en las particiones de prueba según el número de variables que se le
    permiten al modelo. Marca el máximo y el modelo compacto (regla de 1 error estándar).
    """
    ax, own = _own_axes(ax, (7.2, 4.6))
    color = _comparativa_color(comparativa)
    data = curve if k_max_display is None else curve[curve['k'] <= k_max_display]
    best, compact = curve[curve['k'] == k_best].iloc[0], curve[curve['k'] == k_1se].iloc[0]
    ax.plot(data['k'], data['AUC_media'], color=color, lw=2.2, marker='o', markersize=4)
    ax.scatter([k_best], [best['AUC_media']], s=110, facecolor='white', edgecolor=INK, linewidth=1.8, zorder=5,
               label=f"AUC máximo: {k_best} variables")
    ax.scatter([k_1se], [compact['AUC_media']], s=110, color=INK, zorder=6, edgecolor='white', linewidth=1.4,
               label=f"Modelo compacto: {k_1se} variables")
    ax.set_xlabel('Número de variables')
    ax.set_ylabel('AUC')
    ax.legend(loc='lower right')
    ax.set_title(comparativa_label(comparativa))
    return _finish(own, filename or f"Eliminacion_{comparativa}", output_dir, show)

# --- Notebook 03: evaluación de rigor -----------------------------------------------------

def plot_auc_comparison(results, filename='Comparacion_Modelos', output_dir='./img/', show=True):
    """
    Barras agrupadas: AUC media (± DE) de varios modelos en cada comparativa.
    `results` es {(comparativa, modelo): resultados de evaluate_nested}.
    """
    comparativas = list(dict.fromkeys(c for c, _ in results))
    models = list(dict.fromkeys(m for _, m in results))
    labels = {m: _model_label(m) for m in models}
    means = {c: {labels[m]: results[(c, m)]['folds']['AUC'].mean() for m in models} for c in comparativas}
    sds = {c: {labels[m]: results[(c, m)]['folds']['AUC'].std() for m in models} for c in comparativas}
    _, ax = plt.subplots(figsize=(0.75 * len(models) * len(comparativas) + 1.4, 4.8))
    _grouped_bars(ax, list(labels.values()), comparativas, means, COMPARATIVA_COLORS, errors=sds)
    ax.axhline(0.5, color=INK_2, ls='--', lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('AUC')
    ax.legend(handles=[Patch(color=COMPARATIVA_COLORS[c], label=comparativa_label(c)) for c in comparativas]
              + [Line2D([], [], color=INK_2, ls='--', lw=1, label='Azar')], loc='upper center', bbox_to_anchor=(0.5, 1.14),
              ncol=len(comparativas) + 1)
    return save_figure(filename, output_dir, show)

def plot_auc_differences(differences, p_values, filename='Diferencias_AUC_Lineas_Base', output_dir='./img/', show=True):
    """
    Barras agrupadas: diferencia media de AUC (modelo radiómico − modelo comparado) en cada comparativa.
    El asterisco marca p < 0.05. `differences` y `p_values` son {(comparativa, modelo): valor}.
    """
    comparativas = list(dict.fromkeys(c for c, _ in differences))
    models = list(dict.fromkeys(m for _, m in differences))
    labels = {m: _model_label(m) for m in models}
    by_label = {label: m for m, label in labels.items()}
    means = {c: {labels[m]: float(np.mean(differences[(c, m)])) for m in models} for c in comparativas}
    _, ax = plt.subplots(figsize=(0.75 * len(models) * len(comparativas) + 1.4, 4.8))
    positions = _grouped_bars(ax, list(labels.values()), comparativas, means, COMPARATIVA_COLORS, fmt='{:+.3f}')
    for (c, label), x in positions.items():
        if p_values[(c, by_label[label])] < 0.05:
            ax.annotate('*', (x, max(means[c][label], 0)), xytext=(0, 11), textcoords='offset points', ha='center',
                        va='bottom', fontsize=16, fontweight='bold')
    ax.axhline(0, color=INK, lw=1)
    low, high = min(min(v.values()) for v in means.values()), max(max(v.values()) for v in means.values())
    ax.set_ylim(min(low, 0) - 0.03, high + 0.08)
    ax.set_ylabel('Diferencia de AUC')
    ax.legend(handles=[Patch(color=COMPARATIVA_COLORS[c], label=comparativa_label(c)) for c in comparativas]
              + [Line2D([], [], marker='$*$', ls='', color=INK, markersize=9, label='p < 0.05')], loc='upper left')
    return save_figure(filename, output_dir, show)

def plot_permutation_null(null_aucs, observed, p_value, comparativa, filename=None, output_dir='./img/', show=True, ax=None):
    """Histograma del AUC con etiquetas permutadas y línea vertical en el AUC con las etiquetas reales."""
    ax, own = _own_axes(ax, (6.4, 4.4))
    ax.hist(np.asarray(null_aucs), bins=30, color=NEUTRAL, edgecolor='white', linewidth=0.5, label='Etiquetas permutadas')
    p_text = 'p < 0.001' if p_value < 0.0011 else f"p = {p_value:.2f}"
    ax.axvline(observed, color=POSITIVE, lw=2.6, label=f"Modelo real ({p_text})")
    ax.set_xlim(min(0.3, np.min(null_aucs) - 0.02), 1.0)
    ax.set_xlabel('AUC')
    ax.set_ylabel('Permutaciones')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10)
    ax.set_title(comparativa_label(comparativa))
    return _finish(own, filename or f"Perm_{comparativa}", output_dir, show)

def plot_cross_horizon(tablas, filename='Horizontes_Cruzados', output_dir='./img/', show=True):
    """
    Barras agrupadas: porcentaje de imágenes de cada cohorte que cada modelo conversor vs Ctrl clasifica
    como conversoras. La barra rayada es la cohorte que el modelo no usó en su entrenamiento.
    `tablas` es {comparativa: tabla con una fila por cohorte}.
    """
    cohortes = list(COHORT_COLORS)
    values = {c: {} for c in tablas}
    hatches = {}
    for comparativa, tabla in tablas.items():
        for cohorte, fila in tabla.iterrows():
            key = cohorte.split(' ')[0]
            values[comparativa][key] = fila['Clasificados_Conversor_%']
            if 'no vista' in cohorte:
                hatches[(comparativa, key)] = '///'
    _, ax = plt.subplots(figsize=(7.6, 4.8))
    _grouped_bars(ax, cohortes, list(tablas), values, COMPARATIVA_COLORS, fmt='{:.0f} %', hatches=hatches, fontsize=11.5)
    ax.set_xticks(range(len(cohortes)), cohortes, fontsize=12.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Clasificadas como conversoras (%)')
    ax.legend(handles=[Patch(color=COMPARATIVA_COLORS[c], label=f"Modelo {comparativa_label(c)}") for c in tablas]
              + [Patch(facecolor=NEUTRAL, hatch='///', edgecolor='white', linewidth=0, label='Cohorte no vista')], loc='upper left')
    return save_figure(filename, output_dir, show)

def plot_shared_variables(differences, labels, filename='Variables_Compartidas', output_dir='./img/', show=True):
    """
    Barras agrupadas: diferencia media de cada cohorte de conversión con Ctrl, en unidades de DE de Ctrl,
    para las variables que comparten los modelos compactos. `differences` es {cohorte: {variable: valor}}
    y `labels` es {variable: etiqueta}.
    """
    groups = [labels[v] for v in labels]
    values = {c: {labels[v]: d[v] for v in labels} for c, d in differences.items()}
    _, ax = plt.subplots(figsize=(11, 4.8))
    _grouped_bars(ax, groups, list(values), values, COHORT_COLORS, fmt='{:+.2f}', fontsize=11)
    ax.axhline(0, color=INK, lw=1)
    low = min(min(v.values()) for v in values.values()); high = max(max(v.values()) for v in values.values())
    ax.set_ylim(min(low, 0) * 1.3 - 0.1, max(high, 0) * 1.3 + 0.1)
    ax.set_ylabel('Diferencia con Ctrl (DE)')
    ax.legend(handles=[Patch(color=COHORT_COLORS[c], label=c) for c in values], loc='upper right')
    return save_figure(filename, output_dir, show)

def plot_longitudinal_pairs(antes, despues, columns, filename='Longitudinal_Hipocampo', output_dir='./img/', show=True):
    """Barras agrupadas: volumen mediano de cada región en la imagen 2Y y en la imagen 1Y (un año después) de los mismos sujetos."""
    regions = list(columns.values())
    values = {'Imagen 2Y': {name: float(np.median(antes[column])) for column, name in columns.items()},
              'Imagen 1Y (un año después)': {name: float(np.median(despues[column])) for column, name in columns.items()}}
    colors = {'Imagen 2Y': COHORT_COLORS['2Y'], 'Imagen 1Y (un año después)': COHORT_COLORS['1Y']}
    _, ax = plt.subplots(figsize=(6.4, 4.8))
    _grouped_bars(ax, regions, list(values), values, colors, fmt='{:.0f}', fontsize=12)
    ax.set_xticks(range(len(regions)), regions, fontsize=12.5)
    ax.set_ylim(0, max(max(v.values()) for v in values.values()) * 1.3)
    ax.set_ylabel('Volumen mediano (mm³)')
    ax.legend(handles=[Patch(color=color, label=label) for label, color in colors.items()], loc='upper center', ncol=2)
    return save_figure(filename, output_dir, show)

def plot_longitudinal_changes(tabla, names_es, top=15, filename='Longitudinal_Regiones', output_dir='./img/', show=True):
    """Barras: cambio porcentual mediano en un año de las regiones con el cambio más significativo."""
    data = tabla.head(top).sort_values('Cambio_Mediano_%')
    positions = np.arange(len(data))
    _, ax = plt.subplots(figsize=(8.4, 0.4 * len(data) + 1.2))
    ax.barh(positions, data['Cambio_Mediano_%'], color=[POSITIVE if c > 0 else NEGATIVE for c in data['Cambio_Mediano_%']], height=0.68)
    for pos, c in zip(positions, data['Cambio_Mediano_%']):
        ax.annotate(f"{_signed(c, 1)} %", (c, pos), xytext=(5 if c > 0 else -5, 0), textcoords='offset points',
                    ha='left' if c > 0 else 'right', va='center', fontsize=10.5)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(positions, [names_es.get(l, str(l)) for l in data['LabelID']])
    ax.tick_params(axis='y', length=0)
    margin = 0.2 * (data['Cambio_Mediano_%'].max() - data['Cambio_Mediano_%'].min())
    ax.set_xlim(min(data['Cambio_Mediano_%'].min(), 0) - margin, max(data['Cambio_Mediano_%'].max(), 0) + margin)
    ax.set_xlabel('Cambio de volumen en un año (%)')
    ax.grid(axis='y', visible=False)
    return save_figure(filename, output_dir, show)

def plot_effect_sizes(tabla, names_es, top=10, filename='Longitudinal_Efectos', output_dir='./img/', show=True):
    """Barras pareadas por región: tamaño del efecto dentro del sujeto y entre cohortes."""
    data = tabla.head(top).assign(intra=lambda t: t['d_intra'].abs(), entre=lambda t: t['d_entre'].abs()).sort_values('intra')
    positions = np.arange(len(data))
    _, ax = plt.subplots(figsize=(8.4, 0.52 * len(data) + 1.3))
    ax.barh(positions + 0.2, data['intra'], height=0.38, color=HIGHLIGHT, label='Misma persona (2Y → 1Y)')
    ax.barh(positions - 0.2, data['entre'], height=0.38, color=NEUTRAL, label='Personas distintas (cohorte 2Y vs 1Y)')
    ax.set_yticks(positions, [names_es.get(l, str(l)) for l in data['LabelID']])
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Tamaño del efecto (|d|)')
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    return save_figure(filename, output_dir, show)
