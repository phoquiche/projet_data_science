import pandas as pd
import matplotlib.pyplot as plt

# Importation des données
data = pd.read_csv('car_insurance.csv')

# Examen des premières lignes et des informations sur les colonnes
first_rows = data.head()
info = data.info()
description = data.describe()

# Détection des valeurs manquantes
missing_values = data.isna().sum()

# Visualisation des distributions des variables numériques
data.hist(bins=50, figsize=(20, 15))
plt.show()

first_rows, info, description, missing_values

print(missing_values)
