from flask import Flask, request, jsonify, send_from_directory
import mlflow.keras # Usa keras si guardaste con mlflow.keras.log_model
import os

app = Flask(__name__, static_folder="static")

# Ruta del modelo registrado en el MLflow Model Registry
# Ahora que tu modelo SP500_LSTM_Predictor está registrado como la Versión 1
model_uri = "models:/SP500_LSTM_Predictor/1" 

try:
    model = mlflow.keras.load_model(model_uri)
    print(f"Modelo cargado exitosamente desde: {model_uri}")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    # Considera una lógica para manejar el error, por ejemplo, salir o poner el modelo en None

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        # Aquí es donde necesitarás adaptar la entrada para tu modelo LSTM.
        # Tu modelo LSTM fue entrenado con `look_back` pasos de tiempo y 1 feature.
        # y espera una entrada con la forma (1, look_back, 1).
        # Actualmente, tu código espera 'SP500_lag1' y 'SP500_lag2',
        # que son adecuados para modelos como Regresión Lineal o ARIMAX,
        # pero no para el LSTM univariado que entrenaste.

        # Para tu LSTM, necesitarías recibir una secuencia de datos.
        # Por simplicidad en este ejemplo, voy a asumir que recibes una lista de valores
        # y que se escalarán de la misma manera que en el entrenamiento.
        # *** ESTA PARTE ES CRÍTICA: La entrada de 'predict' debe coincidir con la forma y preprocesamiento de 'X_train' ***
        
        # Opciones para la entrada del LSTM:
        # 1. El cliente envía directamente una secuencia de `look_back` valores.
        # 2. El cliente envía `lag1` y `lag2`, pero tu modelo LSTM solo usa 'SP500' univariado.
        #    Esto significa que tu modelo LSTM NO PUEDE usar `lag1` y `lag2` directamente
        #    como fueron generados en el `prepare_data_ts`.
        #    El LSTM fue entrenado para predecir el siguiente valor basado en `look_back` valores ANTERIORES
        #    de la SERIE `SP500` ESCALADA.

        # Adaptación necesaria:
        # 1. Necesitas el 'scaler' que se usó durante el entrenamiento para escalar las entradas.
        #    La forma más robusta es guardarlo junto con el modelo o como un artefacto separado.
        #    Si no lo guardaste, tendrás que recrearlo con los mismos parámetros.
        # 2. La entrada `data` de la solicitud JSON debe proporcionar una *secuencia* de datos,
        #    no solo `lag1` y `lag2`.

        # Para hacer que funcione CON TU CÓDIGO ACTUAL, asumiendo que el modelo de Flask
        # es PARA LOS MODELOS LINEAL/ARIMAX/SARIMAX que sí usan lag1 y lag2:
        # Si este app.py es exclusivamente para el LSTM, debes cambiar cómo recibes la data.

        # Si quieres usar los lags (como en tu código original), tendrías que cargar
        # un modelo que acepte esos lags como features de entrada.
        # Por ejemplo, el modelo Linear Regression o ARIMAX.

        # Si este `app.py` es para el `SP500_LSTM_Predictor`, la lógica de `predict` DEBE CAMBIAR.
        # Asumiendo que `model` es el SP500_LSTM_Predictor:

        # --- Aclaración Importante sobre el `predict` de LSTM ---
        # Tu modelo LSTM fue entrenado para tomar una secuencia de `look_back` valores
        # de `SP500` escalados. No fue entrenado con `SP500_lag1` y `SP500_lag2` como features separados.
        # Por lo tanto, `data['SP500_lag1']` y `data['SP500_lag2']` NO son las entradas correctas
        # para este modelo LSTM.

        # Para el modelo LSTM, la función `predict` esperaría algo como:
        # Una secuencia de `look_back` valores de SP500, escalados.
        # Por ejemplo, si `look_back` es 60, esperarías 60 valores.
        # Ejemplo: data = {"sp500_sequence": [v1, v2, ..., v60]}

        # Dado que tu `predict` actual usa `lag1` y `lag2`, este `app.py` parece estar
        # diseñado para el modelo de Regresión Lineal o ARIMAX/SARIMAX.
        # Si quieres usar el LSTM, tendrías que modificar el `predict` así:

        # (Descomenta y adapta esto si el app.py es para LSTM)
        # from sklearn.preprocessing import MinMaxScaler
        # # Necesitarías instanciar el mismo scaler que se usó en el entrenamiento.
        # # Lo ideal es guardarlo con el modelo o como un artefacto separado.
        # # Por ahora, como ejemplo, lo crearemos aquí (esto no es ideal para producción):
        # # Asumiendo que sabes el rango de los datos originales
        # # (por ejemplo, si SP500 siempre está entre 1000 y 5000)
        # temp_scaler = MinMaxScaler(feature_range=(0, 1))
        # # Necesitas 'fit' el scaler con los mismos datos de entrenamiento que usaste.
        # # Una forma es cargar un dataset dummy o guardar y cargar el scaler.
        # # Para una demostración rápida, si el scaler no ha sido guardado, esto es un placeholder:
        # # Puedes cargarlo de un archivo si lo guardaste en el entrenamiento.
        # # Ejemplo:
        # # import joblib
        # # scaler = joblib.load('path/to/your/scaler.pkl')

        # if "sp500_sequence" not in data:
        #     return jsonify({"error": "Missing 'sp500_sequence' in request data for LSTM model"}), 400
        #
        # raw_sequence = np.array(data["sp500_sequence"]).reshape(-1, 1)
        #
        # # Asegúrate de que el scaler esté ajustado correctamente
        # # Esto es una simulación; en producción, el scaler debe ser persistido y cargado.
        # # Una forma "hacky" para una prueba rápida si no guardaste el scaler:
        # # Necesitas los min y max con los que se ajustó el scaler.
        # # O pasarlos como parte de la solicitud JSON.
        # # Por ejemplo, si entrenaste en datos de SP500 de 1000 a 4000:
        # # Esto NO es robusto para producción, solo para probar si funciona.
        # temp_scaler.fit(np.array([1000.0, 4000.0]).reshape(-1, 1)) # Reemplazar con tus min/max reales
        #
        # scaled_sequence = temp_scaler.transform(raw_sequence)
        #
        # # Reformar para la entrada del LSTM: [1, look_back, 1]
        # if scaled_sequence.shape[0] != model.input_shape[1]: # look_back
        #     return jsonify({"error": f"Expected sequence length of {model.input_shape[1]}, but got {scaled_sequence.shape[0]}"}), 400
        #
        # input_for_prediction = scaled_sequence.reshape(1, model.input_shape[1], 1)
        #
        # prediction_scaled = model.predict(input_for_prediction)
        # prediction_actual = temp_scaler.inverse_transform(prediction_scaled)[0][0]
        # return jsonify({"prediction": float(prediction_actual)})


        # --- Si este app.py es para Linear Regression / ARIMAX / SARIMAX ---
        # Si tu intención es que esta app sirva al modelo de Regresión Lineal, ARIMAX o SARIMAX
        # (que sí usan `SP500_lag1` y `SP500_lag2`), entonces la lógica actual está bien para ellos.
        # Solo asegúrate de cargar el modelo correcto.

        lag1 = float(data['SP500_lag1'])
        lag2 = float(data['SP500_lag2'])
        
        # Para Linear Regression, SARIMAX, ARIMAX que esperan 2 features:
        prediction = model.predict([[lag1, lag2]]) # Esto asume que el modelo cargado acepta [lag1, lag2]

        # Si el modelo cargado (model) es un modelo SARIMAX/ARIMAX, necesitará `.forecast()` y el exog.
        # La línea `prediction = model.predict([[lag1, lag2]])` no funcionaría para SARIMAX/ARIMAX directamente.
        # Para SARIMAX/ARIMAX, necesitarías:
        # model.forecast(steps=1, exog=np.array([[lag1, lag2]]))
        # Y asegurarte de que la data de entrada sea para el exógeno.

        return jsonify({"prediction": float(prediction[0][0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)