import matplotlib.pyplot as plt
import pandas as pd

def plot_history_forecast(history: pd.Series,
                          forecast: pd.Series,
                          actual: pd.Series = None,
                          title: str = '') -> None:
    """Строит график истории, прогноза и, опционально, фактических значений.

    Параметры
    ---------
    history : pd.Series
        Временной ряд исторических данных.
    forecast : pd.Series
        Временной ряд прогнозных значений.
    actual : pd.Series, optional
        Временной ряд фактических значений для сравнения с прогнозом.
        По умолчанию None.
    title : str, optional
        Заголовок графика. По умолчанию пустая строка.

    Возвращает
    ----------
    None
        Функция отображает график и ничего не возвращает.
    """
    plt.figure(figsize=(15,5))
    plt.plot(history.index, history, label='history', alpha=.6)
    plt.plot(forecast.index, forecast, label='forecast', lw=2)
    if actual is not None:
        plt.plot(actual.index, actual, label='actual', lw=2, alpha=.7)
    plt.title(title); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.show()
