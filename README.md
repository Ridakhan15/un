Practical 1 – Data Augmentation
import cv2, numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_digits

img = load_digits().images[0]
img = cv2.resize(img, (64,64))
img = (img * 255 / img.max()).astype(np.uint8)

augmented = {
    'Original' : img,
    'Flip'     : cv2.flip(img, 1),
    'Rotate'   : cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
    'Bright'   : cv2.convertScaleAbs(img, alpha=1.5, beta=30),
    'Noise'    : np.clip(img+np.random.randint(-20,20,img.shape),0,255)
}

fig, axes = plt.subplots(1, 5, figsize=(14,3))
for ax,(t,im) in zip(axes, augmented.items()):
    ax.imshow(im, cmap='gray'); ax.set_title(t); ax.axis('off')
plt.tight_layout(); plt.show()

Practical 2 – Transfer Learning
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# PCA = frozen feature extractor, SVC = new classifier head
model = Pipeline([('pca', PCA(30)), ('clf', SVC())])
model.fit(X_tr[:100], y_tr[:100])   # train on limited data
print('Accuracy:', accuracy_score(y_te, model.predict(X_te)))

Practical 3 – Few Shot Learning
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Take few samples (3 from each class)
X_few = []
y_few = []

for c in np.unique(y):
    idx = np.where(y == c)[0][:3]
    X_few.append(X[idx])
    y_few.append(y[idx])

X_few = np.vstack(X_few)
y_few = np.hstack(y_few)

# Train model
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_few, y_few)

# Predict and check accuracy
pred = model.predict(X)
print("Accuracy:", accuracy_score(y, pred))

Practical 4 – PCA
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = load_breast_cancer()
X, y = data.data, data.target

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# PCA (2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot
plt.scatter(X_pca[:,0], X_pca[:,1], c=y)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Plot")
plt.show()

Practical 5 – SVD
from sklearn.datasets import load_wine
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# Load dataset
data = load_wine()
X, y = data.data, data.target

# Apply SVD
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)

# Plot
plt.scatter(X_svd[:,0], X_svd[:,1], c=y)
plt.xlabel("SVD1")
plt.ylabel("SVD2")
plt.title("SVD Projection")
plt.show()

Practical 6 – K-Means
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X = StandardScaler().fit_transform(load_iris().data)

inertias = [KMeans(k,n_init=10,random_state=42).fit(X).inertia_ for k in range(1,9)]
plt.plot(range(1,9), inertias, '-o'); plt.title('Elbow Method'); plt.show()

labels = KMeans(3, n_init=10, random_state=42).fit_predict(X)
Xp = PCA(2).fit_transform(X)
plt.scatter(Xp[:,0], Xp[:,1], c=labels, cmap='tab10')
plt.title('KMeans (k=3)'); plt.show()

Practical 7 – DBSCAN
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

data = load_iris()
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.8, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

print("Cluster labels:", np.unique(labels))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="tab10", s=40)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("DBSCAN Clustering (Density-Based)")
plt.show()













Practical 8 – Hierarchical Clustering
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

X = StandardScaler().fit_transform(load_iris().data)
dendrogram(linkage(X,'ward'), truncate_mode='level', p=4)
plt.title('Dendrogram'); plt.show()

labels = AgglomerativeClustering(3, linkage='ward').fit_predict(X)
Xp = PCA(2).fit_transform(X)
plt.scatter(Xp[:,0], Xp[:,1], c=labels, cmap='tab10')
plt.title('Hierarchical Clustering'); plt.show()

Practical 9 – Apriori
Practical 9 – Market Basket Analysis using Apriori Algorithm
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

dataset = [
    ['Milk', 'Bread', 'Butter'],
    ['Beer', 'Bread'],
    ['Milk', 'Bread', 'Butter', 'Beer'],
    ['Bread', 'Butter'],
    ['Milk', 'Beer'],
    ['Milk', 'Bread'],
    ['Butter', 'Beer'],
    ['Milk', 'Bread', 'Butter']
]

df = pd.DataFrame(dataset, columns=['Item1', 'Item2', 'Item3', 'Item4'])
print(df)

te = TransactionEncoder()
te_array = te.fit(dataset).transform(dataset)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)
print(df_encoded)

frequent_itemsets = apriori(
    df_encoded,
    min_support=0.3,
    use_colnames=True
)
print(frequent_itemsets)

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6
)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

Practical 10 – FP-Growth Algorithm
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

transactions = [
    ['Fever', 'Cough', 'Headache'],
    ['Fever', 'Cough'],
    ['Cough', 'Shortness of Breath'],
    ['Fever', 'Headache'],
    ['Cough', 'Chest Pain'],
    ['Fever', 'Cough', 'Chest Pain'],
    ['Headache', 'Nausea'],
    ['Fever', 'Cough', 'Shortness of Breath'],
    ['Cough', 'Headache'],
    ['Fever', 'Nausea'],
    ['Chest Pain', 'Shortness of Breath'],
    ['Fever', 'Cough', 'Headache'],
    ['Cough', 'Nausea'],
    ['Fever', 'Chest Pain'],
    ['Cough', 'Shortness of Breath', 'Chest Pain']
]

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
print(df.head())

frequent_itemsets = fpgrowth(
    df,
    min_support=0.3,
    use_colnames=True
)
print(frequent_itemsets.sort_values('support', ascending=False))

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6
)
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
print(rules.sort_values('lift', ascending=False))
