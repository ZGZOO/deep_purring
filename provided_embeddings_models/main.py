from typing import Tuple, Literal

import joblib
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from provided_embeddings_models.constants import *
from provided_embeddings_models.lightning_model import CatMLP
from provided_embeddings_models.preprocessing import split_embeddings, build_preprocessing_pipeline, get_loaders
from provided_embeddings_models.tasks import TaskType, TASK_CONFIGS
from provided_embeddings_models.util import *

pl.seed_everything(RANDOM_STATE, workers=True)

# Task selection ##############################################################
TASK: TaskType = "gender"  # MUST BE "age_group" | "gender" | "age"
cfg = TASK_CONFIGS[TASK]

# Data loading ################################################################
# TODO: Add an end-to-end loading mode here. When enabled, replace load_embeddings_data() with load_audio_data() from
#  audio_loading.py, which runs the full audio -> spectrogram -> CNN -> embedding pipeline.
data = load_embeddings_data(YAMNET_FILENAME)


def get_age_group_split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/val/test sets with 'age_group' label only. (Values categorical [0, 1, 2])

    :return: train_features, val_features, test_features, train_targets, val_targets, test_targets
    """
    age_group_cleaned = clean_embeddings_for_age_group(data)
    return split_embeddings(age_group_cleaned, "age_group", val_size=0.12, regression=False)


def get_age_split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/val/test sets with 'age' label only. (Values continuous 0.0+)

    :return: train_features, val_features, test_features, train_targets, val_targets, test_targets
    """
    age_cleaned = clean_embeddings_for_age(data)
    return split_embeddings(age_cleaned, "age", val_size=0.12, regression=True)


def get_gender_split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/val/test sets with 'gender' label only. (Values categorical [0, 1])
    Unknown gender rows are removed.

    :return: train_features, val_features, test_features, train_targets, val_targets, test_targets
    """
    gender_cleaned = clean_embeddings_for_gender(data)
    return split_embeddings(gender_cleaned, "gender", val_size=0.12, regression=False)


def standardize_features() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Standardize the training, validation, and test sets, fitting to only the training set.

    :return: train_features, val_features, test_features
    """
    scaler = StandardScaler()
    _scaled_train = scaler.fit_transform(raw_train)
    _scaled_val = scaler.transform(raw_val)
    _scaled_test = scaler.transform(raw_test)
    return _scaled_train, _scaled_val, _scaled_test


# Split and scale embeddings ##################################################
if TASK == "age_group":
    raw_train, raw_val, raw_test, train_labels, val_labels, test_labels = get_age_group_split()
elif TASK == "gender":
    raw_train, raw_val, raw_test, train_labels, val_labels, test_labels = get_gender_split()
else:
    raw_train, raw_val, raw_test, train_labels, val_labels, test_labels = get_age_split()

scaled_train, scaled_val, scaled_test = standardize_features()

# Optuna hyperparam search  ###################################################
es_mode: Literal["max", "min"] = "max" if cfg.direction == "maximize" else "min"


def objective(trial: optuna.Trial) -> float:
    """
    Objective function for optuna study.

    Optimize MLP by tuning PCA # components for input, # hidden layers, hidden layer dim, learning rate.
    Optimize for max macro F1-score (classification) or min MAE (regression).
    """
    # TODO: These search grids are a bit narrow and the models perform somewhat poorly, especially on age regression.
    #  Expand them to cover a wider search space and ideally find better hyperparameters for better performance.
    n_components = trial.suggest_int("n_components", 8, 64)  # PCA components
    n_layers = trial.suggest_int("n_layers", 1, 4)  # num hidden layers
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])  # hidden layer dim
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)  # learning rate

    pca_pipe = Pipeline([("pca", PCA(n_components=n_components, random_state=RANDOM_STATE))])
    train_pca = pca_pipe.fit_transform(scaled_train)
    val_pca = pca_pipe.transform(scaled_val)

    _train_loader, _val_loader, _ = get_loaders(
        train_pca, val_pca, val_pca,
        train_labels, val_labels, val_labels,
        batch_size=BATCH_SIZE, regression=cfg.regression,
    )

    model = CatMLP(
        input_size=n_components,
        hidden_sizes=[hidden_size] * n_layers,
        output_size=cfg.output_size,
        lr=lr,
        task=TASK,
    )

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[EarlyStopping(monitor="val_metric", patience=10, mode=es_mode)],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, _train_loader, _val_loader)
    return trainer.callback_metrics["val_metric"].item()


# Find optimal hyperparams
study = optuna.create_study(
    direction=cfg.direction,
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
study.optimize(objective, n_trials=50)

best = study.best_params
print(f"\nBest hyperparameters: {best}")
print(f"Best val metric: {study.best_value:.4f}")

# Build preprocessing pipeline with optimal hyperparams #######################
final_pipeline = build_preprocessing_pipeline(n_components=best["n_components"], random_state=RANDOM_STATE)
train_final = final_pipeline.fit_transform(raw_train)
val_final = final_pipeline.transform(raw_val)
test_final = final_pipeline.transform(raw_test)

# Save for deployment
joblib_path = MODEL_DIR / f"{TASK}_pipeline.joblib"
joblib.dump(final_pipeline, joblib_path)
print(f"Saved preprocessing pipeline to {joblib_path}")

# Train/Eval final model with optimal hyperparams #############################
train_loader, val_loader, test_loader = get_loaders(
    train_final, val_final, test_final,
    train_labels, val_labels, test_labels,
    batch_size=BATCH_SIZE, regression=cfg.regression,
)

final_model = CatMLP(
    input_size=best["n_components"],
    hidden_sizes=[best["hidden_size"]] * best["n_layers"],
    output_size=cfg.output_size,
    lr=best["lr"],
    task=TASK,
)

final_trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[EarlyStopping(monitor="val_metric", patience=10, mode=es_mode)],
)
final_trainer.fit(final_model, train_loader, val_loader)
final_trainer.test(final_model, test_loader)

# TODO: After test, retrieve accumulated predictions from final_model and print/log extended metrics
#  (age_group/gender -> classification report; age -> MAE, RMSE, QWK) as computed in on_test_epoch_end().
#  (See lightning_model.py)

# Save for deployment
ckpt_path = MODEL_DIR / f"{TASK}_model.ckpt"
final_trainer.save_checkpoint(ckpt_path)
print(f"Saved model checkpoint to {ckpt_path}")
