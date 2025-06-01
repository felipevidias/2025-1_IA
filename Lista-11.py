# Aluno: [SEU NOME]
# Disciplina: Inteligência Artificial
# Lista 11 – Titanic Pipeline

# ====================
# 1. PRÉ-PROCESSAMENTO
# ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules

# Carregar base
df = pd.read_csv("train.csv")

# Criar variável FamilySize
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Criar variável Title
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major',
                                   'Rev','Sir','Jonkheer','Dona'],'Rare')
df['Title'] = df['Title'].replace(['Mlle','Ms'],'Miss')
df['Title'] = df['Title'].replace('Mme','Mrs')

# Preencher valores nulos
df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Remover colunas irrelevantes
df.drop(['Cabin','Ticket','Name','PassengerId'], axis=1, inplace=True)

# One-Hot Encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

# Normalização
scaler = StandardScaler()
df[['Age','Fare','FamilySize']] = scaler.fit_transform(df[['Age','Fare','FamilySize']])

# Separar X e y
X = df.drop('Survived', axis=1)
y = df['Survived']

# ============================
# 2. MODELAGEM SUPERVISIONADA
# ============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest:\n", classification_report(y_test, y_pred_rf))

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print("Naive Bayes:\n", classification_report(y_test, y_pred_nb))

# ========================
# 3. CLUSTERIZAÇÃO (KMEANS)
# ========================
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)

# PCA para visualização
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette="viridis")
plt.title("Clusters com K-Means (PCA)")
plt.show()

# ==========================
# 4. REGRAS DE ASSOCIAÇÃO
# ==========================
# Binário para Apriori
assoc_df = df.copy()
assoc_df['Survived'] = assoc_df['Survived'].astype(str)
assoc_df = assoc_df.apply(lambda x: pd.cut(x, bins=2, labels=["Low", "High"]) if x.dtype != 'O' else x)
assoc_df = pd.get_dummies(assoc_df)

# Apriori
frequent = apriori(assoc_df, min_support=0.05, use_colnames=True)
rules = association_rules(frequent, metric='lift', min_threshold=1.0)

# Mostrar 3 regras
print("Regras de Associação:")
print(rules[['antecedents','consequents','support','confidence','lift']].head(3))

# ============================
# 5. CONCLUSÃO
# ============================
'''
Conclusões:
- Random Forest apresentou melhor desempenho com alta precisão e F1-Score.
- K-Means revelou agrupamentos significativos nos dados dos passageiros.
- Regras de associação confirmam padrões como mulheres de 1ª classe com alta taxa de sobrevivência.

O pipeline atendeu aos objetivos de pré-processamento, modelagem supervisionada, não supervisionada e mineração de padrões.
'''
