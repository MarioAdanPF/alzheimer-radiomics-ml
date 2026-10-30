# Alzheimer Radiomics ML

Pipeline de Machine Learning para detectar, en pacientes con Deterioro Cognitivo Leve (DCL), la **conversión a enfermedad de Alzheimer (EA)** a partir de biomarcadores **estructurales y morfológicos** extraídos de resonancias magnéticas procesadas con FreeSurfer y PyRadiomics.

## Filosofía del proyecto

- **Solo morfología macroscópica**: descriptores de forma de PyRadiomics (volumen, área, esfericidad, elongación, diámetros…). No se usan biomarcadores moleculares (amiloide, tau, APOE, LCR) ni métricas de intensidad o textura, que dependen del escáner.
- **IA transparente**: el modelo final es una regresión logística con selección Lasso; cada coeficiente es interpretable.
- **Evaluación sin fuga de datos**: validación cruzada repetida y anidada, agrupada por sujeto. Todo el preprocesamiento dependiente de los datos (ajuste por covariables, escalado, filtros, selección de variables, hiperparámetros y umbral) se ajusta solo con el entrenamiento de cada fold.

## Cohortes y comparativas

Tres cohortes de ADNI:

| Target | Cohorte | Significado |
|---|---|---|
| 0 | Ctrl | DCL estable durante más de 4 años (sMCI) |
| 1 | 1Y | Imagen adquirida 12 meses antes del diagnóstico de EA |
| 2 | 2Y | Imagen adquirida 24 meses antes del diagnóstico de EA |

Un mismo sujeto puede aportar una imagen 1Y y otra 2Y; todas las particiones se agrupan por sujeto. Se modelan `1Y_vs_Ctrl`, `2Y_vs_Ctrl` y `1Y_vs_2Y`.

## Estructura del repositorio

```
01_preparacion_datos.ipynb   # Carga, control de calidad, mapeo de sujetos, preprocesamiento y demografía
02_modelo_y_metricas.ipynb   # Pipeline, validación anidada, modelo final, estabilidad, eliminación de variables, exportación
03_evaluacion_rigor.ipynb    # Líneas base, t-test corregido, permutación, algoritmos, horizontes cruzados, análisis longitudinal

ml_core/                     # Código reutilizado por los notebooks y por los pipelines exportados
├── data_loader.py           # Carga de los CSV de PyRadiomics por cohorte
├── preprocess.py            # Purga de textura y normalización por volumen total segmentado
├── covariates.py            # CovariateResidualizer: ajuste por edad y sexo dentro del fold
├── evaluation.py            # Pipeline (filtros, Lasso, regresión logística) y validación cruzada anidada
├── inference.py             # RadiomicsPipeline: inferencia end-to-end desde el CSV crudo
├── regions.py               # Nombres en español de las regiones de FreeSurfer y de los descriptores de forma
└── visualization.py         # Todas las figuras: gráficas simples (barras, dona, matrices, líneas) con el mínimo de texto

exported_pipelines/          # Modelos finales entrenados (.joblib)
img/                         # Figuras generadas por los notebooks
FreeSurferColorLUT.csv       # Nombres de las regiones de FreeSurfer
```

Los notebooks se ejecutan en orden: 01 genera `datos_procesados/`, que usan 02 y 03. Se publican sin salidas porque estas pueden mostrar datos de participantes; las figuras que generan están en `img/`.

## Datos: no incluidos en este repositorio

Este repositorio **no incluye datos de pacientes**. Los datos provienen de [ADNI](https://adni.loni.usc.edu/) y están sujetos a su Data Use Agreement, que prohíbe la redistribución pública. Para reproducir el pipeline necesitas tu propio acceso a ADNI:

```
subjects/
├── ADNI_1Y/PI<ImageDataID>.csv
├── ADNI_2Y/PI<ImageDataID>.csv
└── MCI_Constant_FreeSurfer/<PTID>.csv
adni_meta/                       # Colecciones de imágenes exportadas de IDA (sujeto, sexo, edad, visita, fecha)
├── MCI_to_AD_1Y.csv
├── MCI_to_AD_2Y.csv
└── ConstantMCI.csv
patients_1_to_2copia.xlsx        # Pares de imágenes 1Y/2Y del mismo sujeto
```

Cada CSV de `subjects/` es la salida de PyRadiomics de una imagen: una fila por región (`LabelID`) con sus métricas. Todas estas rutas, `datos_procesados/` y las tablas con identificadores de pacientes están en `.gitignore`.

## Instalación

Requiere Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate  # o .venv\Scripts\activate en Windows
pip install -r requirements.txt
```

## Uso de un modelo exportado

```python
import joblib
import pandas as pd

pipeline = joblib.load("exported_pipelines/pipeline_1Y_vs_Ctrl.joblib")  # ml_core.inference.RadiomicsPipeline
df_crudo = pd.read_csv("ruta/al/paciente.csv")                            # salida de PyRadiomics de una imagen
resultado = pipeline.predict(df_crudo, edad=74, sexo='F')                  # edad y sexo al escaneo
print(resultado)  # {'PatientID', 'Prediccion_Clase', 'Probabilidad', 'Umbral_Usado'}
print(pipeline.get_feature_importances())
```

El paquete `ml_core` debe ser importable al cargar el pipeline, porque el modelo serializado usa sus transformers.

## Cómo citar

Este código acompaña la tesis de licenciatura en Ingeniería Biomédica (Universidad Autónoma de Zacatecas):

> Bañuelos Frias, D. P. y Pérez Frias, M. A. (2026). *Predicción de progreso de Deterioro Cognitivo Leve a Alzheimer a partir de Resonancia Magnética*. Universidad Autónoma de Zacatecas.

## Reconocimiento de ADNI

Los datos utilizados en la preparación de este trabajo se obtuvieron de la base de datos de la Alzheimer's Disease Neuroimaging Initiative (ADNI) ([adni.loni.usc.edu](https://adni.loni.usc.edu/)). Por ello, los investigadores de ADNI contribuyeron al diseño y a la implementación de ADNI o proporcionaron datos, pero no participaron en el análisis ni en la redacción de este trabajo. La lista completa de los investigadores de ADNI se encuentra en el [ADNI Acknowledgement List](http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf).

La recolección y el intercambio de los datos de ADNI están financiados por el National Institute on Aging (National Institutes of Health, subvención U19AG024904). La organización beneficiaria es el Northern California Institute for Research and Education. Anteriormente, ADNI también recibió financiamiento del National Institute of Biomedical Imaging and Bioengineering, de los Canadian Institutes of Health Research y de contribuciones del sector privado a través de la Foundation for the National Institutes of Health (FNIH), incluidas las generosas aportaciones de: AbbVie, Alzheimer's Association; Alzheimer's Drug Discovery Foundation; Araclon Biotech; BioClinica, Inc.; Biogen; Bristol-Myers Squibb Company; CereSpir, Inc.; Cogstate; Eisai Inc.; Elan Pharmaceuticals, Inc.; Eli Lilly and Company; EuroImmun; F. Hoffmann-La Roche Ltd and its affiliated company Genentech, Inc.; Fujirebio; GE Healthcare; IXICO Ltd.; Janssen Alzheimer Immunotherapy Research & Development, LLC.; Johnson & Johnson Pharmaceutical Research & Development LLC.; Lumosity; Lundbeck; Merck & Co., Inc.; Meso Scale Diagnostics, LLC.; NeuroRx Research; Neurotrack Technologies; Novartis Pharmaceuticals Corporation; Pfizer Inc.; Piramal Imaging; Servier; Takeda Pharmaceutical Company; and Transition Therapeutics.

## Licencia

El código se distribuye bajo la licencia [MIT](LICENSE). La licencia no cubre los datos de ADNI, que se rigen por su propio acuerdo de uso.
