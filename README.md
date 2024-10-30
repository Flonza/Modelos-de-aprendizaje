# Modelos de aprendizaje supervizado

Este proyecto incluye dos carpetas principales: `heart` y `Red Wine Quality`. A continuación se describe el contenido de cada una.

## Heart

Esta carpeta contiene los datos utilizados para entrenar el modelo de inteligencia artificial utilizando el modelo de regresion logistica. Incluye los siguientes archivos:

- `heart_cleveland_upload.csv`: Contiene las características de los datos de entrenamiento.
- `regresion_logistica.py`: Contiene el codigo del entrenamiento.

## Red Wine Quality

Esta carpeta contiene los datos utilizados para evaluar el rendimiento del modelo entrenado por medio del modelo de arboles de decision. Incluye los siguientes archivos:

- `winequality-red.csv`: Contiene las características de los datos de prueba.
- `arboles_decicion.csv`: Contiene el codigo entrenado.

## Cómo Usar Este Proyecto

1. Clona este repositorio: `git clone <URL del repositorio>`
2. Navega a la carpeta del proyecto: `cd <nombre del proyecto>`
3. Revisa las carpetas `heart` y `Red Wine Quality` para familiarizarte con los datos.
4. Ejecuta el script de arboles dentro de la carpeta de red wine: `python arboles_decicion.py`
5. Ejecuta el script de regrecion logistica dentro de la carpeta heart: `python regrecion_logistica.py`

## Requisitos

- Python 3.12.6
- Pandas
- Scikit-learn
