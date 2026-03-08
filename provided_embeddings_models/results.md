## Age Group Classification

Best hyperparameters: {'n_components': 64, 'n_layers': 1, 'hidden_size': 64, 'lr': 0.0002156579828606166}

Best val metric: 0.7463 (macro F1-score)

| Name    | Type              | Params | Mode  | FLOPs |
|---------|-------------------|--------|-------|-------|
| net     | Sequential        | 4.4 K  | train | 0     |
| loss_fn | CrossEntropyLoss  | 0      | train | 0     |
| metric  | MulticlassF1Score | 0      | train | 0     |

* 4.4 K Trainable params
* 0 Non-trainable params
* 4.4 K Total params
* 0.017 Total estimated model params size (MB)
* 6 Modules in train mode
* 0 Modules in eval mode
* 0 Total Flops

test metric: 0.6917523145675659 (macro F1-score)

Saved preprocessing pipeline to models/age_group_pipeline.joblib

Saved model checkpoint to models/age_group_model.ckpt

## Age Regression

Best hyperparameters: {'n_components': 30, 'n_layers': 4, 'hidden_size': 64, 'lr': 0.0001555810423537441}

Best val metric: 2.9848 (MAE)

| Name    | Type              | Params | Mode  | FLOPs |
|---------|-------------------|--------|-------|-------|
| net     | Sequential        | 14.5 K | train | 0     |
| loss_fn | MSELoss           | 0      | train | 0     |
| metric  | MeanAbsoluteError | 0      | train | 0     |

* 14.5 K Trainable params
* 0 Non-trainable params
* 14.5 K Total params
* 0.058 Total estimated model params size (MB)
* 12 Modules in train mode
* 0 Modules in eval mode
* 0 Total Flops

test_metric: 4.101724147796631 (MAE)

Saved preprocessing pipeline to models/age_pipeline.joblib

Saved model checkpoint to models/age_model.ckpt

## Gender Classification

Best hyperparameters: {'n_components': 56, 'n_layers': 2, 'hidden_size': 128, 'lr': 0.00420169371894944}

Best val metric: 0.6897 (macro F1-score)

| Name    | Type             | Params | Mode  | FLOPs |
|---------|------------------|--------|-------|-------|
| net     | Sequential       | 24.1 K | train | 0     |
| loss_fn | CrossEntropyLoss | 0      | train | 0     |   
| metric  | BinaryF1Score    | 0      | train | 0     |

* 24.1 K Trainable params
* 0 Non-trainable params
* 24.1 K Total params
* 0.096 Total estimated model params size (MB)
* 8 Modules in train mode
* 0 Modules in eval mode
* 0 Total Flops

test_metric: 0.6666666865348816 (macro F1-score)

Saved preprocessing pipeline to models/gender_pipeline.joblib

Saved model checkpoint to models/gender_model.ckpt