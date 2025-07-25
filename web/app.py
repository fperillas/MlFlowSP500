from flask import Flask, request, jsonify, send_from_directory
import mlflow.keras  # Usa keras si guardaste con mlflow.keras.log_model
import os

app = Flask(__name__, static_folder="static")

# Ruta local del modelo registrado
model_uri = "mlartifacts/models/m-6247634d9861458bb8b39b3f91df9af0/artifacts"
model = mlflow.keras.load_model(model_uri)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        lag1 = float(data['SP500_lag1'])
        lag2 = float(data['SP500_lag2'])
        prediction = model.predict([[lag1, lag2]])
        return jsonify({"prediction": float(prediction[0][0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

if __name__ == "__main__":
    app.run(debug=True,host = "0.0.0.0", port=3000)