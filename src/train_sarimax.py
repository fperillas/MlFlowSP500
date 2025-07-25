import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import mlflow
import mlflow.statsmodels
import os
from src.utils import prepare_data_ts, get_segment_data # Importar funciones de utilidad

def train_sarimax_model():
    """
    Entrena un modelo SARIMAX para pronosticar el S&P 500
    utilizando el segmento 'middle' del dataset y registra los resultados en MLflow
    bajo el experimento 'sp500_SARIMAX'.
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("sp500_SARIMAX")

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'SP500.csv')
    full_data = prepare_data_ts(data_path)

    if full_data is None:
        return

    # --- Seleccionar el segmento 'middle' de los datos ---
    segment_data = get_segment_data(full_data, 'middle')

    if len(segment_data) < 50: # SARIMAX puede necesitar más datos que ARIMAX para estacionalidad
        print("Error: Pocos datos en el segmento 'middle' para SARIMAX. Se necesita más historia.")
        return

    # Dividir el segmento en conjuntos de entrenamiento y prueba
    train_size = int(len(segment_data) * 0.8)
    train_data = segment_data.iloc[:train_size]
    test_data = segment_data.iloc[train_size:]

    y_train = train_data['SP500']
    exog_train = train_data[['SP500_lag1', 'SP500_lag2']]
    
    y_test = test_data['SP500']
    exog_test = test_data[['SP500_lag1', 'SP500_lag2']]

    with mlflow.start_run():
        print("Iniciando run de MLflow para SARIMAX (Segmento Middle)...")

        # Definir el orden SARIMAX (p, d, q) y (P, D, Q, S)
        # S=5 para estacionalidad semanal (5 días hábiles), común para datos diarios.
        order = (1, 1, 1)
        seasonal_order = (1, 0, 0, 5) # Ejemplo: AR estacional de orden 1, sin diferenciación estacional, MA estacional de orden 0, periodo 5

        # Instanciar y entrenar el modelo SARIMAX
        try:
            model = SARIMAX(y_train, exog=exog_train, order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False)
        except Exception as e:
            print(f"Error al entrenar el modelo SARIMAX: {e}")
            return

        # Realizar predicciones
        y_pred = model_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, exog=exog_test)
        y_pred.index = y_test.index

        # Calcular RMSE
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"RMSE para SARIMAX (Segmento Middle): {rmse}")

        # Registrar parámetros y métricas en MLflow
        mlflow.log_param("model_type", "SARIMAX")
        mlflow.log_param("segment", "middle")
        mlflow.log_param("data_points_in_segment", len(segment_data))
        mlflow.log_param("ar_order", order[0])
        mlflow.log_param("diff_order", order[1])
        mlflow.log_param("ma_order", order[2])
        mlflow.log_param("seasonal_ar_order", seasonal_order[0])
        mlflow.log_param("seasonal_diff_order", seasonal_order[1])
        mlflow.log_param("seasonal_ma_order", seasonal_order[2])
        mlflow.log_param("seasonal_period", seasonal_order[3])
        mlflow.log_metric("rmse", rmse)
        
        # Registrar el modelo de statsmodels
        mlflow.statsmodels.log_model(model_fit, artifact_path="model_sarimax")
        print("Modelo SARIMAX (Segmento Middle) registrado en MLflow.")

        # Generar y guardar la gráfica de Predicción vs Real
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_train.index, y_train, label="Real (Entrenamiento)", color='green', alpha=0.7)
        ax.plot(y_test.index, y_test, label="Real (Prueba)", color='blue')
        ax.plot(y_pred.index, y_pred, label="Predicción (Prueba)", color='orange', linestyle='--')
        ax.set_title("Predicción vs Real (SARIMAX - Segmento Middle)")
        ax.legend()
        plt.tight_layout()

        plot_filename = "pred_vs_real_sarimax_middle.png"
        fig.savefig(plot_filename)
        mlflow.log_artifact(plot_filename)
        print(f"Gráfica '{plot_filename}' registrada en MLflow.")
        plt.close(fig)
        os.remove(plot_filename)

        print("Run de MLflow para SARIMAX (Segmento Middle) completado.")
        print(f"View run at: {mlflow.active_run().info.artifact_uri}")

if __name__ == "__main__":
    train_sarimax_model()