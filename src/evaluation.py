from sklearn.metrics import roc_auc_score
import numpy as np

def latency_gain(y_true, y_pred, timestamps):
    """Вычисляет выигрыш во времени до срабатывания алерта"""
    # предполагаем timestamps, y_true, y_pred одномерные np.array
    # находим первые индексы, где value превышает порог (True for delay)
    # latency_gain = время между предсказанием и фактом
    incident_mask = y_true > 0  # здесь порог уже закодирован во входе
    alert_time = timestamps[np.argmax(y_pred * incident_mask)]
    true_time = timestamps[np.argmax(incident_mask)]
    return (true_time - alert_time).astype('timedelta64[s]').item()

def eval_roc_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)
