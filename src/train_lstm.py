import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import os
from src.utils import prepare_data_ts, get_segment_data # Importar funciones de utilidad

# Función para crear secuencias para LSTM
def create_sequences(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0]) # Features (en este caso, SP500_scaled)
        y.append(data[i + look_back, 0])    # Target (siguiente SP500_scaled)
    return np.array(X), np.array(y)

def train_lstm_model():
    """
    Entrena un modelo LSTM para pronosticar el S&P 500
    utilizando el segmento 'late' del dataset y registra los resultados en MLflow
    bajo el experimento 'sp500_LSTM'.
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("sp500_LSTM")

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'SP500.csv')
    full_data = prepare_data_ts(data_path)

    if full_data is None:
        return

    # --- Seleccionar el segmento 'late' de los datos ---
    segment_data = get_segment_data(full_data, 'late')

    if len(segment_data) < 100: # LSTM generalmente necesita más datos
        print("Error: Pocos datos en el segmento 'late' para LSTM. Se necesita más historia.")
        return

    # Usaremos solo la columna 'SP500' para el LSTM por simplicidad
    # Los lags ya están implícitos en la creación de secuencias para la serie univariada
    # Si quisieras usar 'SP500_lag1' y 'SP500_lag2' como features, la creación de secuencias
    # debería ser multi-variada, lo que añadiría complejidad.
    dataset = segment_data['SP500'].values.reshape(-1, 1)

    # Escalar los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset)

    # Dividir datos en conjuntos de entrenamiento y prueba (cronológicamente)
    train_size = int(len(dataset_scaled) * 0.8)
    train_data_scaled = dataset_scaled[0:train_size,:]
    test_data_scaled = dataset_scaled[train_size:len(dataset_scaled),:]

    # Crear secuencias para LSTM
    look_back = 60 # Número de pasos de tiempo a considerar para la predicción
    X_train, y_train = create_sequences(train_data_scaled, look_back)
    X_test, y_test = create_sequences(test_data_scaled, look_back)

    # Reformar la entrada para LSTM: [samples, time_steps, features]
    # En este caso, 1 feature (SP500)
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Error: No se pudieron crear suficientes secuencias con look_back={look_back}. Intenta reducirlo o usar un segmento más grande.")
        return

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


    with mlflow.start_run():
        print("Iniciando run de MLflow para LSTM (Segmento Late)...")

        # Construir el modelo LSTM
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1)) # Capa de salida para regresión (un solo valor)

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Callbacks para Early Stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Entrenar el modelo
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, 
                            validation_split=0.1, callbacks=[early_stopping], verbose=0)
        print("Modelo LSTM entrenado.")

        # Realizar predicciones
        y_pred_scaled = model.predict(X_test)
        
        # Invertir el escalado para obtener los valores reales
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_inv = scaler.inverse_transform(y_pred_scaled)

        # Calcular RMSE
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        print(f"RMSE para LSTM (Segmento Late): {rmse}")

        # Registrar parámetros y métricas en MLflow
        mlflow.log_param("model_type", "LSTM")
        mlflow.log_param("segment", "late")
        mlflow.log_param("data_points_in_segment", len(segment_data))
        mlflow.log_param("look_back_window", look_back)
        mlflow.log_param("epochs", len(history.epoch)) # Épocas reales entrenadas
        mlflow.log_param("batch_size", 32)
        mlflow.log_metric("rmse", rmse)
        
        # Registrar el modelo de Keras
        mlflow.keras.log_model(model, artifact_path="model_lstm")
        print("Modelo LSTM (Segmento Late) registrado en MLflow.")

        # Generar y guardar la gráfica de Predicción vs Real
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Para graficar, es útil tener el índice de tiempo correcto.
        # Ajustamos los índices de las predicciones para que coincidan con y_test_inv.
        # Usamos el índice de la parte de 'test' del segmento original.
        test_indices = segment_data.index[train_size + look_back : train_size + look_back + len(y_test_inv)]

        ax.plot(segment_data.index[:train_size], scaler.inverse_transform(train_data_scaled).flatten(), 
                label="Real (Entrenamiento)", color='green', alpha=0.7)
        ax.plot(test_indices, y_test_inv, label="Real (Prueba)", color='blue')
        ax.plot(test_indices, y_pred_inv, label="Predicción (Prueba)", color='orange', linestyle='--')
        
        ax.set_title("Predicción vs Real (LSTM - Segmento Late)")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Valor S&P 500")
        ax.legend()
        plt.tight_layout()

        plot_filename = "pred_vs_real_lstm_late.png"
        fig.savefig(plot_filename)
        mlflow.log_artifact(plot_filename)
        print(f"Gráfica '{plot_filename}' registrada en MLflow.")
        plt.close(fig)
        os.remove(plot_filename)

        print("Run de MLflow para LSTM (Segmento Late) completado.")
        print(f"View run at: {mlflow.active_run().info.artifact_uri}")

if __name__ == "__main__":
    train_lstm_model()
