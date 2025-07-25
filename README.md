# 📈 Modelo S&P 500 con MLflow

Este proyecto demuestra el entrenamiento y seguimiento de distintos modelos de **series de tiempo** para la predicción del índice **S&P 500**, utilizando `MLflow` para el manejo de experimentos, métricas, parámetros y versiones de modelos. El dataset utilizado es `SP500.csv`.

---

## 🗂️ Estructura del Proyecto

```
mi_proyecto_ml/
├── data/
│   └── SP500.csv                  # 📊 Dataset histórico del S&P 500
├── notebooks/
│   └── entrenamiento.ipynb       # 📓 Exploración interactiva y visualización
├── src/
│   ├── main_train.py             # 🚀 Script principal de entrenamiento
│   ├── utils.py                  # 🧰 Funciones auxiliares
│   ├── train_arimax.py           # 🔁 Entrenamiento modelo ARIMAX
│   ├── train_sarimax.py          # 🔁 Entrenamiento modelo SARIMAX
│   ├── train_lstm.py             # 🔮 Entrenamiento modelo LSTM
│   └── train_linear_regression.py # 📉 Entrenamiento modelo Regresión Lineal
├── requirements.txt              # 📦 Librerías necesarias
└── README.md                     # 📄 Este archivo
```

---

## ✅ Prerrequisitos

Asegúrate de tener instalado:

- 🐍 [Python 3.8+](https://www.python.org/downloads/)
- 📦 `pip`
- 🔧 [Git](https://git-scm.com/downloads)

---

## ⚙️ Configuración del Entorno

1. **Clona el repositorio**:
   ```bash
   git clone <URL_DE_TU_REPOSITORIO>
   cd mi_proyecto_ml
   ```

2. **Crea y activa un entorno virtual**:
   ```bash
   python -m venv venv_mlflow

   # En Windows:
   .\venv_mlflow\Scripts\activate

   # En macOS/Linux:
   source venv_mlflow/bin/activate
   ```

3. **Instala dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Coloca el dataset**:
   Asegúrate de que `SP500.csv` esté en:
   ```
   mi_proyecto_ml/data/SP500.csv
   ```

---

## ▶️ Cómo Ejecutar el Proyecto

### 1. Iniciar servidor de MLflow:
```bash
mlflow ui
```
Accede en: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 2. Ejecutar script principal:
```bash
python src/main_train.py
```

### 3. Ver resultados:
Desde MLflow UI podrás explorar:

- 🛠️ Parámetros de entrenamiento
- 📊 Métricas (RMSE, MAE, etc.)
- 📁 Artefactos: modelos, gráficas, etc.

---

## 🧠 Modelos Implementados

| Modelo                  | Descripción                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **📉 Regresión Lineal** (`sp500_Linear_Regression`) | Modelo clásico con variables rezagadas |
| **🔁 ARIMAX** (`sp500_ARIMAX`) | Modelo estadístico con variables exógenas, entrenado en el tramo **early** |
| **🌀 SARIMAX** (`sp500_SARIMAX`) | Similar al ARIMAX pero incluye **estacionalidad**, entrenado en el **middle** |
| **🔮 LSTM** (`sp500_LSTM`) | Red neuronal recurrente avanzada, entrenada en el segmento **late** |

---

## 📓 Cuaderno Jupyter

Para exploración y visualización, revisa:

```
notebooks/entrenamiento.ipynb
```

---

## ☁️ Consideraciones para Despliegue en AWS EC2

### 🧮 Elección de Instancia

| Modelo | Instancia Recomendada |
|--------|------------------------|
| Lineal/ARIMAX/SARIMAX | `t3.medium` o similar |
| LSTM (con TensorFlow) | `g4dn.xlarge` o similar (con GPU) |

### 🔐 Seguridad

- Abre el **puerto 22 (SSH)** solo a tu IP.
- Abre el **puerto 5000 (MLflow UI)** si deseas acceso externo (usa seguridad extra en producción).

### 💾 Persistencia

- Usa **Amazon S3** para guardar artefactos de MLflow si deseas que persistan después de apagar la instancia.

### 🔧 En la instancia

```bash
# Instalar dependencias
python -m venv venv_mlflow
source venv_mlflow/bin/activate
pip install -r requirements.txt
```

Si usas GPU:
- Instala drivers NVIDIA + CUDA.
- Usa TensorFlow con soporte GPU.

### 📤 Transferencia de archivos

```bash
scp -i tu_clave.pem -r mi_proyecto_ml/ ec2-user@<IP_PUBLICA_EC2>:/home/ec2-user/
```

---

> Proyecto desarrollado para demostración de buenas prácticas en entrenamiento de modelos de series de tiempo y gestión de experimentos con MLflow.

---
