import pandas as pd
import numpy as np
import joblib
from .preprocess import purge_texture_features, normalize_by_proxy_etiv

class RadiomicsPipeline:
    """
    Pipeline de Inferencia End-to-End para el proyecto de Alzheimer.
    Toma datos crudos de PyRadiomics de un paciente, valida regiones, aplana, normaliza
    por el volumen total segmentado y aplica el modelo final: un Pipeline de sklearn
    (ajuste por covariables, escalado, filtros, selección Lasso y regresión logística)
    con el umbral de decisión elegido en validación cruzada.
    """
    def __init__(self, model, threshold, valid_labels, input_columns, covariates=()):
        self.model = model
        self.threshold = threshold
        self.valid_labels = valid_labels
        self.input_columns = list(input_columns)
        self.covariates = list(covariates)

    def _flatten(self, df_raw):
        # 1. Validación de regiones (LabelIDs)
        missing_labels = set(self.valid_labels) - set(df_raw['LabelID'].unique())
        if missing_labels:
            raise ValueError(f"Faltan las siguientes {len(missing_labels)} regiones requeridas: {sorted(missing_labels)[:5]}...")

        # 2. Filtrado estricto a las regiones del entrenamiento
        df = df_raw[df_raw['LabelID'].isin(self.valid_labels)].drop(columns=['PatientID', 'Target'], errors='ignore')

        # 3. Aplanado (pivot) a una sola fila: Label_<id>_<característica>
        df_long = df.melt(id_vars=['LabelID'], var_name='Caracteristica', value_name='Valor')
        df_long['Columna'] = 'Label_' + df_long['LabelID'].astype(str) + '_' + df_long['Caracteristica']
        df_flat = pd.DataFrame([df_long.set_index('Columna')['Valor'].to_dict()])

        # 4. Mismo preprocesamiento que en el entrenamiento: solo forma y normalización por volumen total
        df_flat = normalize_by_proxy_etiv(purge_texture_features(df_flat).apply(pd.to_numeric, errors='coerce'))

        # 5. Columnas en el mismo orden que en el entrenamiento
        radiomic_columns = [c for c in self.input_columns if c not in self.covariates]
        missing = set(radiomic_columns) - set(df_flat.columns)
        if missing:
            raise ValueError(f"Faltan características requeridas post-procesamiento: {sorted(missing)[:5]}...")
        X = df_flat[radiomic_columns]
        if X.isna().any().any() or np.isinf(X.to_numpy(dtype=float)).any():
            raise ValueError("El paciente tiene valores faltantes o infinitos en características requeridas.")
        return X

    def predict(self, df_raw, edad=None, sexo=None):
        """
        Recibe un DataFrame crudo con formato largo (PatientID, LabelID, características),
        como el generado directamente por FreeSurfer/PyRadiomics. Si el modelo está ajustado
        por covariables, requiere la edad (años) y el sexo ('F' o 'M') al momento del escaneo.
        """
        patient_id = df_raw['PatientID'].iloc[0] if 'PatientID' in df_raw.columns else "Paciente_X"
        X = self._flatten(df_raw)

        if self.covariates:
            if edad is None or sexo not in ('F', 'M'):
                raise ValueError("Este modelo está ajustado por edad y sexo: indique edad (años) y sexo ('F' o 'M').")
            X = X.assign(Edad=float(edad), Sexo_F=int(sexo == 'F'))
        X = X[self.input_columns]

        prob = float(self.model.predict_proba(X)[0, 1])
        return {
            'PatientID': patient_id,
            'Prediccion_Clase': int(prob >= self.threshold),
            'Probabilidad': prob,
            'Umbral_Usado': self.threshold
        }

    def get_feature_importances(self):
        """
        Coeficientes de la regresión logística (sobre características escaladas con RobustScaler),
        ordenados por magnitud. Positivo: favorece la clase conversora.
        """
        classifier = self.model.named_steps['clasificador']
        df_imp = pd.DataFrame({'Caracteristica': classifier.feature_names_in_, 'Peso': classifier.coef_[0]})
        return df_imp.reindex(df_imp['Peso'].abs().sort_values(ascending=False).index).reset_index(drop=True)

def export_pipeline(model, threshold, valid_labels, input_columns, output_path, covariates=()):
    """
    Crea una instancia de RadiomicsPipeline y la guarda en disco.
    """
    pipeline = RadiomicsPipeline(
        model=model,
        threshold=threshold,
        valid_labels=valid_labels,
        input_columns=input_columns,
        covariates=[c for c in covariates if c in input_columns]
    )
    joblib.dump(pipeline, output_path)
    print(f"Pipeline guardado en: {output_path}")
    return pipeline
