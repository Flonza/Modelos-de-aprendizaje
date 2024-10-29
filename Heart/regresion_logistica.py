# Importacion de librerias necesarias para el análisis y la modelacion
import pandas as pd                      # Manejo de datos
import numpy as np                       # Operaciones numéricas
import matplotlib.pyplot as plt          # Gráficos generales
import seaborn as sns                    # Visualizacion avanzada de datos
from sklearn.model_selection import train_test_split, GridSearchCV  # Division de datos y optimizacion de hiperparámetros
from sklearn.linear_model import LogisticRegression                 # Modelo de Regresion Logistica
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc  # Métricas de evaluacion
from sklearn.feature_selection import SelectKBest, f_classif        # Seleccion de caracteristicas

# Cargar datos del archivo CSV
datos = pd.read_csv("heart_cleveland_upload.csv")  # El DataFrame 'datos' contiene los datos cargados

# Exploracion inicial de los datos
print(datos.head())      # Muestra las primeras filas para ver la estructura
print(datos.info())      # Muestra informacion de columnas, tipos de datos y valores nulos
print(datos.describe())  # Estadisticas descriptivas básicas para cada columna numérica

# Visualizacion de la distribucion de la variable objetivo 'condition' (condicion cardiaca)
sns.countplot(data=datos, x='condition')  
plt.title("Distribucion de Enfermedades Cardiacas (0: No, 1: Si)")
plt.show()

# Verificacion y eliminacion de valores nulos (vacios)
print("Valores nulos por columna:\n", datos.isnull().sum())  # Cuenta los valores nulos en cada columna
datos.dropna(inplace=True)  # Elimina las filas con valores nulos

# Transformacion de variables categoricas en variables dummy (0 o 1)
datos = pd.get_dummies(datos, drop_first=True)  # Drop_first evita la multicolinealidad al eliminar una categoria de referencia

# Separacion de las variables predictoras (X) y la variable objetivo (y)
X = datos.drop('condition', axis=1)  # 'X' contiene todas las variables excepto 'condition'
y = datos['condition']               # 'y' contiene únicamente la variable objetivo 'condition'

# Seleccion de caracteristicas usando el método SelectKBest
selector_caracteristicas = SelectKBest(f_classif, k=5)  # Selecciona las 5 caracteristicas más relevantes
X_seleccionado = selector_caracteristicas.fit_transform(X, y)
caracteristicas_seleccionadas = X.columns[selector_caracteristicas.get_support()]
print("Caracteristicas seleccionadas:", caracteristicas_seleccionadas)

# Division de datos en conjunto de entrenamiento y prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X[caracteristicas_seleccionadas], y, test_size=0.3, random_state=42
)

# Ajuste de hiperparametros del modelo mediante búsqueda en cuadricula (GridSearchCV)
parametros = {'C': [0.1, 1, 10, 100], 'penalty': ['l2']}  # Definicion del espacio de busqueda
busqueda_cuadricula = GridSearchCV(LogisticRegression(), parametros, cv=5)  # Validacion cruzada de 5 pliegues
busqueda_cuadricula.fit(X_entrenamiento, y_entrenamiento)
print("Mejores hiperparámetros:", busqueda_cuadricula.best_params_)

# Realizaciin de predicciones y evaluacion del modelo
predicciones = busqueda_cuadricula.predict(X_prueba)
print("Exactitud:", accuracy_score(y_prueba, predicciones))  # Imprime la exactitud del modelo
print("Reporte de clasificacion:\n", classification_report(y_prueba, predicciones))  # Muestra la precision y recall

# Matriz de Confusion para analizar errores de clasificacion
matriz_confusion = confusion_matrix(y_prueba, predicciones)
sns.heatmap(matriz_confusion, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Matriz de Confusion")
plt.xlabel("Prediccion")
plt.ylabel("Real")
plt.show()

# Curva ROC para evaluar el rendimiento del modelo en distintos umbrales
probabilidades_pred = busqueda_cuadricula.predict_proba(X_prueba)[:, 1]  # Probabilidades de prediccion para la clase positiva
falsos_positivos, verdaderos_positivos, umbrales = roc_curve(y_prueba, probabilidades_pred)
roc_auc = auc(falsos_positivos, verdaderos_positivos)

# Gráfico de la Curva ROC
plt.figure()
plt.plot(falsos_positivos, verdaderos_positivos, color='darkorange', lw=2, label='Curva ROC (área = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Linea de referencia para un clasificador aleatorio
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()
