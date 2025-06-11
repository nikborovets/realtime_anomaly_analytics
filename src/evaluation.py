import torch
import sklearn.metrics as skm

def regression_metrics(y_true, y_pred):
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    mse = torch.mean((y_true - y_pred) ** 2).item()
    return {"mae": mae, "mse": mse}


def roc_auc_alert(errors, labels):
    # errors: numpy, labels: 0/1
    return skm.roc_auc_score(labels, errors)