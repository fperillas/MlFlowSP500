from flask import Flask, request, jsonify, send_from_directory
import mlflow.keras
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# Puedes importar joblib si guardaste tu scaler con joblib
# import joblib 

app = Flask(__name__, static_folder="static")

# Configurar la URI de seguimiento de MLflow (útil si vas a hacer otras operaciones MLflow)
# Aunque para cargar directamente con 'file://' no es estrictamente necesario,
# es buena práctica si tu aplicación podría interactuar con el servidor de tracking de otra manera.
mlflow.set_tracking_uri("http://localhost:5000") 

# --- Ruta DIRECTA al modelo LSTM en el sistema de archivos ---
# Necesitas REEMPLAZAR 'TU_EXPERIMENT_ID' y 'TU_RUN_ID' con los valores reales
# que provienen del run que registró tu SP500_LSTM_Predictor.
# Basado en la imagen de tu UI (wistful-bird-281), los valores son:
# Experiment ID: 758453172600072499
# Run ID:        4f8a12e100d74d51a0a94be5c5afc
# Y el artifact_path que usaste para loggear el modelo es "model_lstm"

# Asegúrate de que esta ruta ABSOLUTA sea correcta en tu sistema EC2
# Por ejemplo, si app.py está en ~/MlFlowSP500/web/ y mlruns/ está dentro de 'web'
# la ruta sería: /home/ubuntu/MlFlowSP500/web/mlruns/...
# # Obtiene el directorio de app.py
#model_relative_path = "mlruns/models/SP500_LSTM_Predictor/version-1"
#model_uri = f"file://{os.path.join(base_path, model_relative_path)}"
model_uri ="mlartifacts/758453172600072499/models/m-4e2b8579356e4247ae670b0a32d1dbad/artifacts"
# Si el scaler también fue loggeado como un artefacto y quieres cargarlo de esa manera:
#model_uri = "mlruns/758453172600072499/4f8a12e100d74d51a0a94be5c5afc/artifacts/scaler.pkl"
#scaler_uri = f"file://{os.path.join(base_path, scaler_relative_path)}"


model = None
scaler = None

try:
    model = mlflow.keras.load_model(model_uri)
    print(f"Modelo LSTM cargado exitosamente desde: {model_uri}")

    # *** PUNTO CRÍTICO: El scaler. ***
    # La forma ideal es que el scaler se haya guardado como un artefacto con el modelo
    # o de forma independiente y que lo cargues aquí.
    # Por ejemplo, si lo guardaste en el run del entrenamiento:
    # scaler = joblib.load(mlflow.artifacts.download_artifacts(scaler_uri))
    # Esto descargaría el artefacto si mlflow.set_tracking_uri() está configurado.

    # PARA UNA DEMOSTRACIÓN RÁPIDA (SI NO GUARDASTE EL SCALER):
    # Necesitas saber los valores min y max del dataset de entrenamiento que se usaron para ajustar el scaler.
    # AJUSTA ESTOS VALORES A TUS DATOS REALES DE ENTRENAMIENTO de la columna 'SP500'.
    # Si tu SP500 real iba, por ejemplo, de 1000 a 4000:
    scaler_min = 1000.0 # <--- REEMPLAZA CON EL VALOR MÍNIMO REAL DE SP500 EN TUS DATOS DE ENTRENAMIENTO
    scaler_max = 4000.0 # <--- REEMPLAZA CON EL VALOR MÁXIMO REAL DE SP500 EN TUS DATOS DE ENTRENAMIENTO
    
    # Creamos un scaler y lo "ajustamos" con los min/max conocidos.
    # Esto es un workaround. La mejor práctica es guardar y cargar el scaler.
    dummy_data_for_scaler = np.array([scaler_min, scaler_max]).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(dummy_data_for_scaler)

    print(f"Scaler inicializado (min={scaler_min}, max={scaler_max}). (Verifica estos valores)")

except Exception as e:
    print(f"Error fatal al cargar el modelo o inicializar el scaler: {e}")
    model = None
    scaler = None
    # No llamar a exit() aquí para permitir que el servidor Flask inicie y muestre 404 para predict.
    # El usuario podrá ver que la app está corriendo pero la predicción fallará.

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Modelo o scaler no cargados correctamente en el servidor. Revise los logs de inicio."}), 500

    data = request.json
    try:
        # La entrada para el modelo LSTM debe ser una secuencia de SP500, no solo lags.
        # Asumimos que el cliente envía una lista de los últimos `look_back` valores de SP500.
        # El `look_back` de tu modelo es 60 (asumido por el entrenamiento LSTM).
        if "sp500_sequence" not in data:
            return jsonify({"error": "Se requiere 'sp500_sequence' en el JSON para el modelo LSTM."}), 400

        raw_sequence = np.array(data["sp500_sequence"]).reshape(-1, 1)

        # Verificar que la longitud de la secuencia sea la esperada por el modelo (look_back)
        # Esto asume que el input_shape del modelo está correctamente definido (ej. (None, 60, 1))
        # Si tu modelo tiene model.input_shape[1] = 60
        expected_look_back = model.input_shape[1] 
        if raw_sequence.shape[0] != expected_look_back:
            return jsonify({
                "error": f"La longitud de la secuencia ('sp500_sequence') debe ser {expected_look_back}, pero se recibió {raw_sequence.shape[0]}."
            }), 400

        # Escalar la secuencia de entrada
        scaled_sequence = scaler.transform(raw_sequence)

        # Reformar la entrada para el LSTM: [samples, time_steps, features] -> [1, look_back, 1]
        input_for_prediction = scaled_sequence.reshape(1, expected_look_back, 1)

        # Realizar la predicción
        prediction_scaled = model.predict(input_for_prediction)

        # Invertir el escalado de la predicción para obtener el valor real
        prediction_actual = scaler.inverse_transform(prediction_scaled)[0][0]

        return jsonify({"prediction": float(prediction_actual)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/")
def index():
    # Asume que 'index.html' y 'style.css' están dentro de la carpeta 'static'.
    # Ya corregimos esto previamente moviéndolos a 'static/'.
    return send_from_directory("static", "index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)