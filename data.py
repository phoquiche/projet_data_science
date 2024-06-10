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
data = data.drop(columns=['id'])
data = data.drop(columns=['outcome'])
data = data.drop(columns=['children'])
data.hist(bins=50, figsize=(20, 15), color='#ddfa12', ec='black')
plt.show()

first_rows, info, description, missing_values

print(missing_values)
