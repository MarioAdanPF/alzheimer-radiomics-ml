# Módulos del Core de Machine Learning
from .data_loader import NOMBRES_COHORTE, cargar_cohorte
from .preprocess import TEXTURE_FAMILIES, purge_texture_features, normalize_by_proxy_etiv
from .regions import DESCRIPTOR_ES, region_name_es, load_region_names, split_feature, feature_label_es
from .covariates import CovariateResidualizer
from .evaluation import (
    META_COLUMNS, COVARIATES, METRICS, THRESHOLDS, GRID_LR,
    CorrelationFilter, L1Selector, build_pipeline, selected_features, prepare_comparison,
    best_threshold, classification_metrics, outer_splits, tune_in_fold, evaluate_nested,
    bootstrap_auc_ci, summarize,
)
from .inference import RadiomicsPipeline, export_pipeline
from .visualization import (
    comparativa_label, save_figure,
    plot_cohort_distribution, plot_region_prevalence, plot_invalid_volumes, plot_feature_families,
    plot_total_volume, plot_volume_vs_total, plot_pca_distribution, plot_pca_comparison, plot_demographics,
    plot_pipeline_diagram, plot_nested_cv_diagram, plot_fold_anatomy, plot_auc_by_fold, plot_metrics_panel,
    plot_mean_roc, plot_oof_confusion, plot_coefficients, plot_feature_elimination,
    plot_auc_comparison, plot_auc_differences, plot_permutation_null, plot_cross_horizon, plot_shared_variables,
    plot_longitudinal_pairs, plot_longitudinal_changes, plot_effect_sizes,
)
