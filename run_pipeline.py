# Script reproducible (ejecutar en entorno con requirements)
# Guarda artefactos en outputs/
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
column_names = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                'stalk-surface-below-ring', 'stalk-color-above-ring',
                'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
                'ring-type', 'spore-print-color', 'population', 'habitat']

# Cargar
print('Cargando dataset...')
df = pd.read_csv(url, header=None, names=column_names)

# Imputar stalk-root (opción segura)
df2 = df.copy()
df2['stalk-root'] = df2['stalk-root'].replace('?', np.nan)
imputer = SimpleImputer(strategy='most_frequent')
df2[['stalk-root']] = imputer.fit_transform(df2[['stalk-root']])

# One-hot
X = pd.get_dummies(df2.drop('class', axis=1), dtype=int)
y = df2['class'].map({'e':0,'p':1}).values

# Train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=RANDOM_STATE)

# PCA para visualización
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)
pd.DataFrame(X_pca, columns=['PC1','PC2']).to_csv(os.path.join(OUTDIR,'X_pca.csv'), index=False)

# KMeans k=3
kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init='auto').fit(X_pca)
labels = kmeans.labels_
pd.Series(labels, name='kmeans_k3').to_csv(os.path.join(OUTDIR,'kmeans_k3_labels.csv'), index=False)

# RandomForest
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
joblib.dump(rf, os.path.join(OUTDIR,'rf_model.joblib'))

# Importancias
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
importances.head(50).to_csv(os.path.join(OUTDIR,'rf_top_importances.csv'))

# Predicciones tests
y_pred = rf.predict(X_test)
pd.DataFrame({'y_true':y_test, 'y_pred':y_pred}).to_csv(os.path.join(OUTDIR,'rf_test_preds.csv'), index=False)

print('Pipeline ejecutado. Artefactos en:', OUTDIR)