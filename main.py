import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar los datos
data = pd.read_csv('datos.csv')

# Definir las variables
X = data[['clima', 'precio', 'produccion_anterior']]  # Variables independientes
y = data['demanda']  # Variable dependiente

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Mostrar resultados
print("Error cuadrático medio:", mean_squared_error(y_test, y_pred))

# Ejemplo de predicción con nuevos datos
nuevos_datos = np.array([[25, 100, 1500]])  # Clima, Precio, Producción Anterior
prediccion = model.predict(nuevos_datos)
print("Demanda prevista:", prediccion[0])
