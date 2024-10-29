# Importación de libreras necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif

# Cargar el dataset
df = pd.read_csv("winequality-red.csv") 

# Anaalisis exploratorio de datos
print(df.head())
print(df.info())  
print(df.describe())

# Visualizar la distribucion de la calidad del vino
sns.countplot(data=df, x='quality')
plt.title("Distribución de Calidad del Vino")
plt.show()

# Verificar valores nulos y eliminarlos
print("Valores nulos por columna:\n", df.isnull().sum())
df.dropna(inplace=True)

# Convertir variables categoricas en variables dummy si fuera necesario
# df = pd.get_dummies(df, drop_first=True)  # Aplicar si hay variables categoricas en el dataset

# Separar caracteristicas (variables de entrada) y variable objetivo (calidad del vino)
X = df.drop('quality', axis=1)  
y = df['quality']

# Seleccion de caracteristicas (utilizando SelectKBest)
selector = SelectKBest(f_classif, k=5)  # Selección de las 5 mejores caracteristicas
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("Caracteristicas seleccionadas:", selected_features)

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.3, random_state=42)

# Ajuste de hiperparametros usando GridSearchCV para Árbol de Decision
param_grid = {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 10, 20]}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Mejores hiperparametros:", grid_search.best_params_)

# Predicciones y evaluacion
y_pred = grid_search.predict(X_test)
print("Exactitud:", accuracy_score(y_test, y_pred))
print("Reporte de clasificacion:\n", classification_report(y_test, y_pred))

# Matriz de Confusion
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Matriz de Confusion")
plt.xlabel("Prediccion")
plt.ylabel("Real")
plt.show()

# Importancia de las caracteriticas
feature_importances = grid_search.best_estimator_.feature_importances_
importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Importancia de las Características")
plt.show()
