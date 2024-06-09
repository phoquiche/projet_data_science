from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('car_insurance.csv')

# Examen des premières lignes et des informations sur les colonnes
first_rows = data.head()
info = data.info()
description = data.describe()

# Détection des valeurs manquantes

# Imputation des valeurs manquantes
imputer = SimpleImputer(strategy='median')
data['credit_score'] = imputer.fit_transform(data[['credit_score']])
data['annual_mileage'] = imputer.fit_transform(data[['annual_mileage']])

# Limitation des valeurs aberrantes
data['speeding_violations'] = data['speeding_violations'].apply(lambda x: min(x, 10))

# Encodage des variables qualitatives
categorical_features = ['driving_experience', 'education', 'income', 'vehicle_year', 'vehicle_type']
encoder = OneHotEncoder()

# Transformation des données
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), ['age', 'credit_score', 'annual_mileage']),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Séparation des données en variables explicatives et cible
X = data.drop(['id', 'outcome'], axis=1)
y = data['outcome']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

print(X_train.shape, X_test.shape)
missing_values = data.isna().sum()

# Visualisation des distributions des variables numériques


first_rows, info, description, missing_values

# Création d'un DataFrame avec les données transformées pour les variables numériques uniquement
numeric_data = data[['age', 'credit_score', 'annual_mileage', 'speeding_violations', 'duis', 'past_accidents', 'outcome']]

# Calcul des corrélations
correlation_matrix = numeric_data.corr()

# Visualisation de la matrice de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matrice de Corrélation')
plt.show()
