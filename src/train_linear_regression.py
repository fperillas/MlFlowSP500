import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import os
from src.utils import prepare_data_ts # Importar la función de utilidad para datos de series de tiempo

def train_general_linear_regression_model():
    """
    Entrena un modelo de regresión lineal para pronosticar el S&P 500
    utilizando el dataset completo y registra los resultados en MLflow
    bajo el experimento 'sp500_Linear_Regression'.
    """
    # Configurar la URI de seguimiento de MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Establecer el nombre del experimento de MLflow
    mlflow.set_experiment("sp500_Linear_Regression")

    # Ruta al archivo de datos
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'SP500.csv')
    
    # Cargar y preprocesar los datos usando la función de utilidad para series de tiempo
    # Esta función ya devuelve el DataFrame con los lags y sin NaNs.
    full_data = prepare_data_ts(data_path)

    if full_data is None:
        return # Salir si no se pudieron cargar o preprocesar los datos

    # Dividir datos en características (X) y variable objetivo (y)
    X = full_data[['SP500_lag1', 'SP500_lag2']]
    y = full_data['SP500']
    
    # Dividir datos en conjuntos de entrenamiento y prueba
    # Usamos random_state para reproducibilidad
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Iniciar un nuevo run de MLflow
    with mlflow.start_run():
        print("Iniciando run de MLflow para Linear Regression (General)...")

        # Instanciar y entrenar el modelo de regresión lineal
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Realizar predicciones
        y_pred = model.predict(X_test)

        # Calcular RMSE
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"RMSE para Linear Regression (General): {rmse}")

        # Registrar parámetros y métricas en MLflow
        mlflow.log_param("model_type", "LinearRegression_General")
        mlflow.log_param("coef_lag1", model.coef_[0])
        mlflow.log_param("coef_lag2", model.coef_[1])
        mlflow.log_param("intercept", model.intercept_)
        mlflow.log_metric("rmse", rmse)
        
        # Registrar el modelo de scikit-learn
        mlflow.sklearn.log_model(model, artifact_path="model_linear_regression_general")
        print("Modelo Linear Regression (General) registrado en MLflow.")

        # Generar y guardar la gráfica de Predicción vs Real
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test.values, label="Real", color='blue')
        ax.plot(y_pred, label="Predicción", color='orange')
        ax.set_title("Predicción vs Real (Linear Regression - General)")
        ax.legend()
        plt.tight_layout()

        # Guardar la figura en un archivo temporal para logearla en MLflow
        plot_filename = "pred_vs_real_linear_regression_general.png"
        fig.savefig(plot_filename)
        mlflow.log_artifact(plot_filename)
        print(f"Gráfica '{plot_filename}' registrada en MLflow.")
        plt.close(fig) # Cerrar la figura para liberar memoria
        os.remove(plot_filename) # Eliminar el archivo temporal

        print("Run de MLflow para Linear Regression (General) completado.")
        print(f"View run at: {mlflow.active_run().info.artifact_uri}")

if __name__ == "__main__":
    train_general_linear_regression_model()