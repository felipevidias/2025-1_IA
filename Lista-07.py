import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import seaborn as sns

# Carregar dados
df = pd.read_csv("Iris.csv")
X = df.drop(columns=["class"])

# Outliers (IQR)
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
outlier_mask = ~((X >= (Q1 - 1.5 * IQR)) & (X <= (Q3 + 1.5 * IQR))).all(axis=1)
df_clean = df[~outlier_mask]
X_clean = df_clean.drop(columns=["class"])

# Normalização
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_clean)

# KMeans - Elbow + Silhouette
inertias = []
silhouettes = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

# Gráficos
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(K, inertias, marker='o')
plt.title("Método Elbow")
plt.xlabel("k")
plt.ylabel("Inércia")

plt.subplot(1,2,2)
plt.plot(K, silhouettes, marker='o', color='green')
plt.title("Pontuação Silhouette")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig("grafico_elbow_silhouette.png")

# Melhor K
best_k = silhouettes.index(max(silhouettes)) + 2
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_clean["kmeans_label"] = kmeans.fit_predict(X_scaled)

# Davies-Bouldin
db_score = davies_bouldin_score(X_scaled, df_clean["kmeans_label"])

# DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=4)
df_clean["dbscan_label"] = dbscan.fit_predict(X_scaled)

# SOM
som = MiniSom(x=1, y=3, input_len=4, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_scaled)
som.train_batch(X_scaled, 100)
df_clean["som_label"] = [som.winner(x)[1] for x in X_scaled]

# Erros KMeans
df_clean["original_class"] = df_clean["class"]
df_clean["kmeans_class"] = df_clean["kmeans_label"].map(
    lambda x: df_clean[df_clean["kmeans_label"] == x]["class"].mode()[0]
)
df_clean["erro"] = df_clean["original_class"] != df_clean["kmeans_class"]

# Visualização
plt.figure(figsize=(7,5))
sns.scatterplot(
    x=X_scaled[:, 0], y=X_scaled[:, 1],
    hue=df_clean["erro"],
    palette={True: "red", False: "green"}
)
plt.title("Agrupamento Incorreto (KMeans)")
plt.savefig("erros_kmeans.png")

# Relatório
with open("relatorio.txt", "w") as f:
    f.write("Relatório - Lista 7 - Agrupamento\\n\\n")
    f.write("Melhor k (KMeans): %d\\n" % best_k)
    f.write("Silhouette Score: %.3f\\n" % max(silhouettes))
    f.write("Davies-Bouldin Score: %.3f\\n" % db_score)
    f.write("Clusters pelo DBSCAN: %s\\n" % df_clean["dbscan_label"].nunique())
    f.write("Clusters pelo SOM: %s\\n" % df_clean["som_label"].nunique())
    f.write("\\nAgrupamentos incorretos (KMeans): %d\\n" % df_clean["erro"].sum())
    f.write("Porcentagem correta: %.2f%%\\n" % (100 * (1 - df_clean["erro"].sum() / len(df_clean))))
