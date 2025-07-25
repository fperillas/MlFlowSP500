from flask import Flask, request, jsonify, send_from_directory
import mlflow.sklearn
import os

app = Flask(__name__, static_folder="static")

# Cargar modelo desde MLflow
model_uri = "runs:/75988bc9e3e44bfa80e1bbcb190bb0aa/model_lstm"
model = mlflow.sklearn.load_model(model_uri)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        lag1 = float(data['SP500_lag1'])
        lag2 = float(data['SP500_lag2'])
        prediction = model.predict([[lag1, lag2]])
        return jsonify({"prediction": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Servir el HTML (frontend)
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

if __name__ == "__main__":
    app.run(debug=True)