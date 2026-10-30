"""
Núcleo de evaluación compartido por los notebooks de modelado (02) y de rigor (03).

Contiene solo lo que se reutiliza o lo que se serializa dentro de los pipelines exportados:
los transformers propios del pipeline, su construcción y la validación cruzada anidada.
Los análisis que se usan una sola vez (permutación, t-test corregido, estabilidad,
eliminación de variables, análisis longitudinal…) están escritos en los notebooks.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .covariates import CovariateResidualizer

META_COLUMNS = ['PatientID', 'Target', 'Sujeto_Real']
COVARIATES = ['Edad', 'Sexo_F']
THRESHOLDS = np.linspace(0.1, 0.9, 81)
METRICS = ['AUC', 'Exactitud_Balanceada', 'Sensibilidad', 'Especificidad', 'F1_Macro']

# Grid de hiperparámetros por defecto: C de la selección Lasso (L1) y C de la regresión
# logística final. C de L1 empieza en 0.03 porque con valores menores Lasso deja de
# retener características en muchos folds.
GRID_LR = {
    'seleccion__C': [0.03, 0.05, 0.1, 0.2, 0.5],
    'clasificador__C': [0.01, 0.1, 1.0],
}

class CorrelationFilter(TransformerMixin, BaseEstimator):
    """
    Elimina cada característica cuya correlación absoluta con alguna característica
    anterior supere `threshold`. Se ajusta solo con los datos de `fit` (train del fold).
    """
    def __init__(self, threshold=0.85):
        self.threshold = threshold

    def fit(self, X, y=None):
        corr = np.abs(np.corrcoef(np.asarray(X, dtype=float), rowvar=False))
        self.support_ = ~(np.triu(corr, k=1) > self.threshold).any(axis=0)
        self.feature_names_in_ = np.asarray(X.columns if hasattr(X, 'columns') else range(X.shape[1]), dtype=object)
        return self

    def transform(self, X):
        return X.loc[:, self.support_] if hasattr(X, 'loc') else X[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_[self.support_]

class L1Selector(TransformerMixin, BaseEstimator):
    """
    Selección Lasso: conserva las características con coeficiente no nulo en una regresión
    logística L1. Si Lasso no retiene ninguna (p. ej. con etiquetas permutadas), conserva
    la característica con mayor correlación absoluta con la etiqueta, para que el
    clasificador siempre reciba al menos una.
    """
    def __init__(self, C=0.1):
        self.C = C

    def fit(self, X, y):
        lasso = LogisticRegression(penalty='l1', solver='liblinear', C=self.C,
                                   class_weight='balanced', random_state=42).fit(X, y)
        self.support_ = np.abs(lasso.coef_[0]) > 1e-5
        if not self.support_.any():
            Xc = np.asarray(X, dtype=float) - np.asarray(X, dtype=float).mean(axis=0)
            yc = np.asarray(y, dtype=float) - np.mean(y)
            corr = np.abs(Xc.T @ yc) / (np.linalg.norm(Xc, axis=0) * np.linalg.norm(yc) + 1e-12)
            self.support_[np.argmax(corr)] = True
        self.feature_names_in_ = np.asarray(X.columns if hasattr(X, 'columns') else range(X.shape[1]), dtype=object)
        return self

    def transform(self, X):
        return X.loc[:, self.support_] if hasattr(X, 'loc') else X[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_[self.support_]

def build_pipeline(covariates=COVARIATES, reference_class=None, selection=True,
                   correlation_filter=True, classifier=None, C_l1=0.1, C_lr=0.1):
    """
    Pipeline completo, ajustado de principio a fin solo con el train de cada fold:
    ajuste por covariables -> filtro de varianza -> RobustScaler -> filtro de correlación
    -> selección Lasso (L1) -> clasificador (regresión logística por defecto).
    """
    steps = []
    if covariates:
        steps.append(('ajuste', CovariateResidualizer(tuple(covariates), reference_class)))
    steps += [('varianza', VarianceThreshold(1e-8)), ('escalado', RobustScaler())]
    if correlation_filter:
        steps.append(('correlacion', CorrelationFilter(0.85)))
    if selection:
        steps.append(('seleccion', L1Selector(C_l1)))
    if classifier is None:
        classifier = LogisticRegression(C=C_lr, class_weight='balanced', max_iter=5000)
    steps.append(('clasificador', classifier))
    return Pipeline(steps).set_output(transform='pandas')

def selected_features(model):
    """Nombres de las características que llegan al clasificador de un pipeline ajustado."""
    return list(model.named_steps['clasificador'].feature_names_in_)

def prepare_comparison(df, demografia, clases, columns=None, covariates=COVARIATES):
    """
    Filtra las dos clases de una comparativa y devuelve (X, y, groups).
    y = 1 para la clase con Target mayor (el conversor frente a Ctrl; 2Y en 1Y vs 2Y).
    X contiene las características radiómicas (o `columns`) más las `covariates`
    (que el pipeline consume en el paso de ajuste; usar covariates=None sin ajuste).
    """
    mask = df['Target'].isin(clases)
    features = columns if columns is not None else [c for c in df.columns if c not in META_COLUMNS]
    X = pd.concat([df.loc[mask, features], demografia.loc[mask, list(covariates or [])]], axis=1)
    y = (df.loc[mask, 'Target'] == max(clases)).astype(int).to_numpy()
    groups = df.loc[mask, 'Sujeto_Real'].to_numpy()
    return X.reset_index(drop=True), y, groups

def best_threshold(y, probs):
    """Umbral (0.1–0.9) que maximiza el F1-macro en las probabilidades dadas."""
    scores = [f1_score(y, (probs >= t).astype(int), average='macro', zero_division=0) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(scores))])

def classification_metrics(y, probs, threshold):
    preds = (probs >= threshold).astype(int)
    return {
        'AUC': roc_auc_score(y, probs),
        'Exactitud_Balanceada': balanced_accuracy_score(y, preds),
        'Sensibilidad': recall_score(y, preds, pos_label=1, zero_division=0),
        'Especificidad': recall_score(y, preds, pos_label=0, zero_division=0),
        'F1_Macro': f1_score(y, preds, average='macro', zero_division=0),
    }

def outer_splits(y, groups, n_repeats=5, n_folds=5, seed=0):
    """Particiones externas reproducibles: lista de (repetición, fold, idx_train, idx_test)."""
    splits = []
    for r in range(n_repeats):
        outer_cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
        for f, (idx_train, idx_test) in enumerate(outer_cv.split(np.zeros(len(y)), y, groups)):
            splits.append((r, f, idx_train, idx_test))
    return splits

def tune_in_fold(pipeline, param_grid, X_train, y_train, groups_train, seed):
    """
    Validación interna dentro del train de un fold externo: elige hiperparámetros por AUC
    (GridSearchCV agrupado por sujeto) y el umbral por F1-macro sobre predicciones fuera
    de fold del propio train. El test externo nunca participa.
    Devuelve (modelo ajustado con todo el train, hiperparámetros, umbral).
    """
    inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    if param_grid:
        search = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=1)
        search.fit(X_train, y_train, groups=groups_train)
        model, params = search.best_estimator_, search.best_params_
    else:
        model, params = clone(pipeline).fit(X_train, y_train), {}
    probs_inner = cross_val_predict(clone(model), X_train, y_train, groups=groups_train,
                                    cv=inner_cv, method='predict_proba')[:, 1]
    return model, params, best_threshold(y_train, probs_inner)

def _evaluate_fold(pipeline, param_grid, X, y, groups, idx_train, idx_test, seed):
    model, params, threshold = tune_in_fold(pipeline, param_grid, X.iloc[idx_train], y[idx_train],
                                            groups[idx_train], seed)
    probs = model.predict_proba(X.iloc[idx_test])[:, 1]
    return {'probs': probs, 'threshold': threshold, 'params': params,
            'n_features': len(selected_features(model))}

def evaluate_nested(X, y, groups, pipeline, param_grid=None, n_repeats=5, n_folds=5, seed=0, n_jobs=-1):
    """
    Validación cruzada repetida y anidada agrupada por sujeto.
    Externa: StratifiedGroupKFold (n_folds) x n_repeats, para estimar el desempeño.
    Interna (tune_in_fold): hiperparámetros y umbral elegidos solo con el train externo.
    Las mismas `seed`, `y` y `groups` producen las mismas particiones, así que dos modelos
    evaluados así se pueden comparar fold a fold.

    Devuelve un dict con:
      'folds'     DataFrame con una fila por fold externo (métricas, umbral, hiperparámetros)
      'probs_oof' matriz (n_repeats, n) con la probabilidad fuera de fold de cada imagen
      'y', 'groups', 'splits'
    """
    splits = outer_splits(y, groups, n_repeats, n_folds, seed)
    outputs = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_fold)(pipeline, param_grid, X, y, groups, idx_train, idx_test, seed + 100 * r + f)
        for r, f, idx_train, idx_test in splits)

    folds, probs_oof = [], np.full((n_repeats, len(y)), np.nan)
    for (r, f, idx_train, idx_test), out in zip(splits, outputs):
        probs_oof[r, idx_test] = out['probs']
        folds.append({'Repeticion': r, 'Fold': f, 'n_train': len(idx_train), 'n_test': len(idx_test),
                      'Umbral': out['threshold'], 'n_caracteristicas': out['n_features'],
                      'Hiperparametros': out['params'],
                      **classification_metrics(y[idx_test], out['probs'], out['threshold'])})
    return {'folds': pd.DataFrame(folds), 'probs_oof': probs_oof, 'y': y, 'groups': groups, 'splits': splits}

def bootstrap_auc_ci(y, probs, groups, n_boot=2000, seed=0, alpha=0.05):
    """IC por bootstrap del AUC, remuestreando sujetos (no imágenes)."""
    rng = np.random.default_rng(seed)
    subjects = np.unique(groups)
    indices = {s: np.flatnonzero(groups == s) for s in subjects}
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([indices[s] for s in rng.choice(subjects, len(subjects), replace=True)])
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], probs[idx]))
    return np.quantile(aucs, [alpha / 2, 1 - alpha / 2])

def summarize(results, name):
    """Media ± DE por métrica sobre los folds externos, más IC 95 % del AUC por bootstrap."""
    folds = results['folds']
    row = {'Modelo': name}
    for metric in METRICS:
        row[metric] = f"{folds[metric].mean():.3f} ± {folds[metric].std():.3f}"
    # Probabilidad media por imagen sobre las repeticiones (cada imagen es test una vez por repetición)
    probs_mean = np.nanmean(results['probs_oof'], axis=0)
    ci = bootstrap_auc_ci(results['y'], probs_mean, results['groups'])
    row['AUC_IC95'] = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
    row['Caracteristicas (mediana)'] = int(folds['n_caracteristicas'].median())
    return row
