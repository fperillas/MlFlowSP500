import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import mlflow
import mlflow.statsmodels # Asegúrate de que esta librería esté instalada
import os
from src.utils import prepare_data_ts, get_segment_data # Importar funciones de utilidad

def train_arimax_model():
    """
    Entrena un modelo ARIMAX para pronosticar el S&P 500
    utilizando el segmento 'early' del dataset y registra los resultados en MLflow
    bajo el experimento 'sp500_ARIMAX'.
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("sp500_ARIMAX")

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'SP500.csv')
    full_data = prepare_data_ts(data_path)

    if full_data is None:
        return

    # --- Seleccionar el segmento 'early' de los datos ---
    segment_data = get_segment_data(full_data, 'early')

    if len(segment_data) < 10: # Asegurar que haya suficientes datos para entrenar y probar
        print("Error: Pocos datos en el segmento 'early' para ARIMAX. Se necesita más historia.")
        return

    # Dividir el segmento en conjuntos de entrenamiento y prueba
    # Para series de tiempo, la división debe ser cronológica
    train_size = int(len(segment_data) * 0.8)
    train_data = segment_data.iloc[:train_size]
    test_data = segment_data.iloc[train_size:]

    y_train = train_data['SP500']
    exog_train = train_data[['SP500_lag1', 'SP500_lag2']]
    
    y_test = test_data['SP500']
    exog_test = test_data[['SP500_lag1', 'SP500_lag2']]


    # Definir el nombre del modelo registrado para ARIMAX
    # Asegúrate de que sea un nombre diferente al del modelo LSTM si ambos se van a registrar
    registered_model_name_arimax = "SP500_ARIMAX_Predictor" 

    with mlflow.start_run():
        print("Iniciando run de MLflow para ARIMAX (Segmento Early)...")

        # Definir el orden ARIMA (p, d, q) y sin orden estacional para ARIMAX
        # (p=orden AR, d=diferenciación, q=orden MA)
        # Valores de ejemplo, que podrían ajustarse con análisis de ACF/PACF.
        order = (5, 1, 0) # AR(5), 1 diferenciación, MA(0)

        # Instanciar y entrenar el modelo SARIMAX (se usa para ARIMAX también)
        try:
            model = SARIMAX(y_train, exog=exog_train, order=order,
                            enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False) # disp=False para no mostrar resultados detallados durante el fit
        except Exception as e:
            print(f"Error al entrenar el modelo ARIMAX: {e}")
            return # Salir del run si falla el entrenamiento

        # Realizar predicciones
        # start y end deben ser índices del conjunto de datos original o del test_data
        # Si usas predict, asegúrate de pasar los exógenos correctos para el futuro
        # Para simplificar, haremos una predicción de 'in-sample' para los datos de prueba
        y_pred = model_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, exog=exog_test)
        
        # Asegurarse de que y_pred tenga el mismo índice que y_test para la comparación
        y_pred.index = y_test.index

        # Calcular RMSE
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"RMSE para ARIMAX (Segmento Early): {rmse}")

        # Registrar parámetros y métricas en MLflow
        mlflow.log_param("model_type", "ARIMAX")
        mlflow.log_param("segment", "early")
        mlflow.log_param("data_points_in_segment", len(segment_data))
        mlflow.log_param("ar_order", order[0])
        mlflow.log_param("diff_order", order[1])
        mlflow.log_param("ma_order", order[2])
        mlflow.log_metric("rmse", rmse)
        
        # *** CAMBIO CLAVE AQUÍ: AÑADIR registered_model_name ***
        mlflow.statsmodels.log_model(
            statsmodels_model=model_fit, 
            artifact_path="model_arimax",
            registered_model_name=registered_model_name_arimax # <--- ¡Este es el cambio!
        )
        print(f"Modelo ARIMAX (Segmento Early) registrado como '{registered_model_name_arimax}' en MLflow.")

        # Generar y guardar la gráfica de Predicción vs Real
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_train.index, y_train, label="Real (Entrenamiento)", color='green', alpha=0.7)
        ax.plot(y_test.index, y_test, label="Real (Prueba)", color='blue')
        ax.plot(y_pred.index, y_pred, label="Predicción (Prueba)", color='orange', linestyle='--')
        ax.set_title("Predicción vs Real (ARIMAX - Segmento Early)")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Valor S&P 500")
        ax.legend()
        plt.tight_layout()

        plot_filename = "pred_vs_real_arimax_early.png"
        fig.savefig(plot_filename)
        mlflow.log_artifact(plot_filename)
        print(f"Gráfica '{plot_filename}' registrada en MLflow.")
        plt.close(fig)
        os.remove(plot_filename)

        print("Run de MLflow para ARIMAX (Segmento Early) completado.")
        print(f"View run at: {mlflow.active_run().info.artifact_uri}")

if __name__ == "__main__":
    train_arimax_model()