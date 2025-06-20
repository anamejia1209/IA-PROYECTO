# src/main.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import clean_heart_data
from regression_model import train_regression_model
from logistic_model import train_logistic_model  # ⬅️Linea Nueva

# Cargar datos
df = pd.read_csv('./data/heart_disease_raw.csv')

# Limpiar datos
df_clean = clean_heart_data(df)

# Crear variable binaria (0 = sin enfermedad, 1 = con enfermedad)
df_clean['target_bin'] = df_clean['num'].apply(lambda x: 0 if x == 0 else 1)  # ⬅️Linea Nueva

# Guardar versión limpia
df_clean.to_csv('./data/heart_disease_clean.csv', index=False)

print(df_clean.head())

# Visualizar la distribución de las variables numéricas
df_clean.hist(bins=20, figsize=(14, 10), color='skyblue')
plt.suptitle('Distribuciones de variables numéricas')
plt.tight_layout()
plt.show()

# Visualizar la Matriz de correlaciones
plt.figure(figsize=(12, 10))
correlation = df_clean.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Matriz de correlaciones')
plt.show()

# Conteo de clases
# ⬇️Linea Modificada
sns.countplot(data=df_clean, x='target_bin', hue='target_bin', palette='pastel', legend=False)
plt.title('Distribución de clases (ausencia vs presencia de enfermedad cardíaca)')
plt.xlabel('Enfermedad cardíaca (0 = No, 1 = Sí)')
plt.ylabel('Número de pacientes')
plt.show()

# Entrenamiento del modelo de regresión
train_regression_model(df_clean)

# Entrenamiento del modelo de regresión logística
train_logistic_model(df_clean)  # ⬅️Linea Nueva
