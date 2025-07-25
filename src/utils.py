import pandas as pd
import numpy as np
import os

def prepare_data_ts(data_path):
    """
    Carga y preprocesa los datos del S&P 500 para modelos de series de tiempo.
    Retorna el DataFrame completo con características rezagadas y sin NaNs.

    Args:
        data_path (str): Ruta al archivo CSV del S&P 500.

    Returns:
        pd.DataFrame: DataFrame preprocesado con 'DATE' como índice,
                      'SP500', 'SP500_lag1', 'SP500_lag2'.
                      Retorna None si el archivo no se encuentra.
    """
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de datos en {data_path}")
        print("Asegúrate de que 'SP500.csv' esté en la carpeta 'data' de tu proyecto.")
        return None

    # Preprocesamiento de datos
    data['DATE'] = pd.to_datetime(data['DATE'])
    data.set_index('DATE', inplace=True)

    # Reemplazar '.' con NaN y convertir a float
    data['SP500'] = data['SP500'].replace('.', np.nan).astype(float)

    # Imputar valores nulos con el método forward fill
    data['SP500'] = data['SP500'].ffill()

    # Crear características (variables rezagadas)
    data['SP500_lag1'] = data['SP500'].shift(1)
    data['SP500_lag2'] = data['SP500'].shift(2)
    data.dropna(inplace=True) # Eliminar filas con NaN introducidos por el shift

    return data

def get_segment_data(data_df, segment_name):
    """
    Divide el DataFrame en segmentos cronológicos.

    Args:
        data_df (pd.DataFrame): El DataFrame completo preprocesado.
        segment_name (str): 'early', 'middle', o 'late'.

    Returns:
        pd.DataFrame: El segmento de datos correspondiente.
    """
    total_len = len(data_df)
    
    if segment_name == 'early':
        start_idx = 0
        end_idx = int(total_len * 0.33)
    elif segment_name == 'middle':
        start_idx = int(total_len * 0.33)
        end_idx = int(total_len * 0.66)
    elif segment_name == 'late':
        start_idx = int(total_len * 0.66)
        end_idx = total_len
    else:
        raise ValueError("segment_name debe ser 'early', 'middle' o 'late'")

    segment_data = data_df.iloc[start_idx:end_idx]
    
    if len(segment_data) == 0:
        print(f"Advertencia: El segmento '{segment_name}' está vacío o tiene muy pocos datos.")
    
    return segment_data
