import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class CovariateResidualizer(TransformerMixin, BaseEstimator):
    """
    Elimina de cada característica el efecto lineal de las covariables (p. ej. edad y sexo).

    Los coeficientes se estiman solo con los datos que recibe `fit` (el train de cada fold)
    y, si se indica `reference_class`, solo con esa clase: así no se resta señal de la
    enfermedad cuando la covariable está desbalanceada entre clases (enfoque habitual en
    neuroimagen: estimar el efecto de la edad en el grupo de referencia).

    Recibe un DataFrame que incluye las columnas de covariables y devuelve las
    características ajustadas, sin las covariables. Las covariables se centran, así que
    cada característica conserva su valor esperado en la covariable media.
    """
    def __init__(self, covariates=('Edad', 'Sexo_F'), reference_class=None):
        self.covariates = covariates
        self.reference_class = reference_class

    def fit(self, X, y=None):
        covariates = list(self.covariates)
        if self.reference_class is None:
            mask = np.ones(len(X), dtype=bool)
        else:
            if y is None:
                raise ValueError("reference_class requiere y en fit.")
            mask = np.asarray(y) == self.reference_class

        C = X.loc[mask, covariates].to_numpy(dtype=float)
        F = X.loc[mask].drop(columns=covariates).to_numpy(dtype=float)
        self.covariate_means_ = C.mean(axis=0)
        design = np.column_stack([np.ones(len(C)), C - self.covariate_means_])
        coefs, *_ = np.linalg.lstsq(design, F, rcond=None)
        self.coef_ = coefs[1:]
        self.feature_names_out_ = X.columns.drop(covariates)
        return self

    def transform(self, X):
        check_is_fitted(self, 'coef_')
        C = X[list(self.covariates)].to_numpy(dtype=float) - self.covariate_means_
        F = X[self.feature_names_out_].to_numpy(dtype=float) - C @ self.coef_
        return pd.DataFrame(F, columns=self.feature_names_out_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)
