import numpy as np
import pandas as pd
from datetime import datetime 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Chargement des données à partir du fichier CSV
df = pd.read_csv('sphist.csv')

# Conversion de la colonne 'Date' en objet de type datetime
df['Date'] = pd.to_datetime(df['Date'])

# Filtrage des données pour inclure uniquement les dates supérieures au 1er avril 2015
date_filter = df['Date'] > datetime(year=2015, month=4, day=1)
df = df[date_filter]

# Tri du dataframe par ordre croissant de la colonne 'Date'
df = df.sort_values('Date', ascending=True)

# Calcul des moyennes mobiles sur 5 jours et 365 jours pour la colonne 'Open'
df['5_days_mean'] = df['Open'].rolling(window=5).mean().shift(1)
df['365_days_mean'] = df['Open'].rolling(window=365).mean().shift(1)

# Calcul des écarts types mobiles sur 5 jours et 365 jours pour la colonne 'Open'
df['5_days_std'] = df['Open'].rolling(window=5).std().shift(1)
df['365_days_std'] = df['Open'].rolling(window=365).std().shift(1)

# Calcul des ratios entre les moyennes mobiles sur 5 jours et 365 jours, et entre les écarts types mobiles sur 5 jours et 365 jours
df['5_365_ratio_mean'] = df['5_days_mean'] / df['365_days_mean']
df['5_365_ratio_std'] = df['5_days_std'] / df['365_days_std']

# Extraction des composantes 'Year', 'Month' et 'Day' à partir de la colonne 'Date'
df['Year'] = df['Date'].apply(lambda x: x.year)
df['Month'] = df['Date'].apply(lambda x: x.month)
df['Day'] = df['Date'].apply(lambda x: x.day)

# Filtrage des données pour inclure uniquement les dates supérieures ou égales au 3 janvier 1951
date_filter = df['Date'] >= datetime(year=1951, month=1, day=3)
df = df[date_filter]

# Suppression des lignes contenant des valeurs manquantes
df.dropna(axis=0, inplace=True)

# Séparation des données en ensembles d'entraînement et de test en utilisant la colonne 'Date'
train = df[df['Date'] < datetime(year=2013, month=1, day=1)]
test = df[df['Date'] >= datetime(year=2013, month=1, day=1)]

# Modèle de régression linéaire
lr_model = LinearRegression()
features = ['5_days_mean']
lr_model.fit(train[features], train['Close'])
lr_pred = lr_model.predict(test[features])

# Calcul de l'erreur quadratique moyenne (MSE) pour le modèle de régression linéaire
mse_lr = mean_squared_error(test['Close'], lr_pred)
print("MSE (Linear Regression):", mse_lr)

# Modèle des k plus proches voisins
knn_model = KNeighborsRegressor(n_neighbors=2)
knn_model.fit(train[features], train['Close'])
knn_pred = knn_model.predict(test[features])

# Calcul de l'erreur quadratique moyenne (MSE) pour le modèle des k plus proches voisins
mse_knn = mean_squared_error(test['Close'], knn_pred)
print("MSE (kNN):", mse_knn)

# Ajout de la moyenne mobile sur 5 jours du volume
df['volume_5_day_mean'] = df['Volume'].rolling(window=5).mean().shift(1)

# Suppression des lignes contenant des valeurs manquantes
df.dropna(axis=0, inplace=True)

# Séparation des données en ensembles d'entraînement et de test
train = df[df['Date'] < datetime(year=2013, month=1, day=1)]
test = df[df['Date'] >= datetime(year=2013, month=1, day=1)]

# Modèle de régression linéaire avec la nouvelle caractéristique
lr_model = LinearRegression()
features = ['volume_5_day_mean']
lr_model.fit(train[features], train['Close'])
lr_pred = lr_model.predict(test[features])

# Calcul de l'erreur quadratique moyenne (MSE) pour le modèle de régression linéaire avec la nouvelle caractéristique
mse_lr_new = mean_squared_error(test['Close'], lr_pred)
print("MSE (Linear Regression with new feature):", mse_lr_new)

# Modèle des k plus proches voisins avec la nouvelle caractéristique
knn_model = KNeighborsRegressor(n_neighbors=2)
knn_model.fit(train[features], train['Close'])
knn_pred = knn_model.predict(test[features])

# Calcul de l'erreur quadratique moyenne (MSE) pour le modèle des k plus proches voisins avec la nouvelle caractéristique
mse_knn_new = mean_squared_error(test['Close'], knn_pred)
print("MSE (kNN with new feature):", mse_knn_new)
