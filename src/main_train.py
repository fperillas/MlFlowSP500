import os
from src.train_arimax import train_arimax_model
from src.train_sarimax import train_sarimax_model
from src.train_lstm import train_lstm_model
from src.train_linear_regression import train_general_linear_regression_model # Importar el nuevo script

def run_all_trainings():
    """
    Ejecuta el entrenamiento de todos los modelos (Regresión Lineal general,
    ARIMAX, SARIMAX, LSTM). Cada entrenamiento se registra en su propio experimento de MLflow.
    """
    print("Iniciando el entrenamiento de todos los modelos...")

    # Ejecutar el entrenamiento del modelo de Regresión Lineal (General)
    print("\n--- Entrenando Modelo de Regresión Lineal (General) ---")
    train_general_linear_regression_model()

    # Ejecutar el entrenamiento del modelo ARIMAX (Segmento Early)
    print("\n--- Entrenando Modelo ARIMAX (Segmento Early) ---")
    train_arimax_model()

    # Ejecutar el entrenamiento del modelo SARIMAX (Segmento Middle)
    print("\n--- Entrenando Modelo SARIMAX (Segmento Middle) ---")
    train_sarimax_model()

    # Ejecutar el entrenamiento del modelo LSTM (Segmento Late)
    print("\n--- Entrenando Modelo LSTM (Segmento Late) ---")
    train_lstm_model()

    print("\nTodos los entrenamientos han finalizado.")
    print("Para ver los resultados en la UI de MLflow, ejecuta 'mlflow ui' en tu terminal y navega a http://127.0.0.1:5000.")

if __name__ == "__main__":
    run_all_trainings()