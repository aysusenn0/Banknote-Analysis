import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

column_names = ["variance", "skewness", "curtosis", "entropy", "class"]
dosya_yolu = r"C:\Users\AYSU\OneDrive - İstanbul Medeniyet Üniversitesi\Masaüstü\veri.csv"

df = pd.read_csv(dosya_yolu, header=None, names=column_names)

X = df.drop("class", axis=1)
y = df["class"]

print("\n--- VERİ SETİ ÖZETİ ---")
print(f"Toplam Örnek Sayısı: {len(df)}")
print(f"Özellik Sayısı: {X.shape[1]}")
print("\nSınıf Dağılımı:")
print(y.value_counts(normalize=True))

# Histogramlar
X.hist(figsize=(10, 6), bins=20, edgecolor="black")
plt.suptitle("Özellik Dağılımları")
plt.show()

models = {
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", random_state=42, C=1, gamma="scale"))
    ]),
    "kNN": Pipeline([
        ("model", KNeighborsClassifier(n_neighbors=1))
    ]),
    "Naive Bayes": Pipeline([
        ("model", GaussianNB())
    ])
}

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1"]

results = []

print("\n--- MODEL KARŞILAŞTIRMASI ---")
for name, model in models.items():
    cv = cross_validate(model, X, y, cv=kf, scoring=scoring)
    results.append({
        "Model": name,
        "Accuracy": np.mean(cv["test_accuracy"]),
        "Precision": np.mean(cv["test_precision"]),
        "Recall": np.mean(cv["test_recall"]),
        "F1": np.mean(cv["test_f1"])
    })

results_df = pd.DataFrame(results).set_index("Model")
print(results_df)

# Grafik
results_df[["Accuracy", "F1"]].plot(kind="bar", ylim=(0.8, 1.0), figsize=(8, 5))
plt.title("Model Performans Karşılaştırması")
plt.ylabel("Skor")
plt.grid(axis="y", linestyle="--")
plt.show()

confusion_matrices = {}

for model_name, model in models.items():
    y_pred = cross_val_predict(model, X, y, cv=kf)
    cm = confusion_matrix(y, y_pred)
    confusion_matrices[model_name] = cm

    # Görselleştirme
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# PCA (3 boyut)
pca = PCA(n_components=3)
X_pca_3d = pca.fit_transform(X)

# SVM modeli
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1, gamma="scale"))
])

svm.fit(X_pca_3d, y)

# Support vector'ları al
support_vectors = svm.named_steps["svm"].support_vectors_

# 3D çizim
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

# Normal noktalar
ax.scatter(
    X_pca_3d[y == 0, 0],
    X_pca_3d[y == 0, 1],
    X_pca_3d[y == 0, 2],
    label="Sahte",
    alpha=0.4
)

ax.scatter(
    X_pca_3d[y == 1, 0],
    X_pca_3d[y == 1, 1],
    X_pca_3d[y == 1, 2],
    label="Gerçek",
    alpha=0.4
)

# Support vector'lar (VURGULU)
ax.scatter(
    support_vectors[:, 0],
    support_vectors[:, 1],
    support_vectors[:, 2],
    color="black",
    s=80,
    label="Support Vector"
)

ax.set_xlabel("PCA Bileşen 1")
ax.set_ylabel("PCA Bileşen 2")
ax.set_zlabel("PCA Bileşen 3")
ax.set_title("SVM – Support Vector'lar ile 3B Ayrım Mantığı")
ax.legend()

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# PCA (3 boyut)
pca = PCA(n_components=3)
X_pca_3d = pca.fit_transform(X)

# kNN modeli
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_pca_3d, y)

# Rastgele bir test noktası seç
idx = np.random.randint(len(X_pca_3d))
test_point = X_pca_3d[idx].reshape(1, -1)

# En yakın komşular
distances, neighbors = knn.kneighbors(test_point)

# 3D çizim
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

# Tüm noktalar
ax.scatter(X_pca_3d[y == 0, 0], X_pca_3d[y == 0, 1], X_pca_3d[y == 0, 2],
           alpha=0.3, label="Sahte")
ax.scatter(X_pca_3d[y == 1, 0], X_pca_3d[y == 1, 1], X_pca_3d[y == 1, 2],
           alpha=0.3, label="Gerçek")

# Test noktası
ax.scatter(test_point[0, 0], test_point[0, 1], test_point[0, 2],
           color="black", s=100, label="Test Noktası")

# Komşular
ax.scatter(X_pca_3d[neighbors[0], 0],
           X_pca_3d[neighbors[0], 1],
           X_pca_3d[neighbors[0], 2],
           color="red", s=80, label="En Yakın Komşular")

ax.set_title("kNN – 3B Komşuluk Mantığı (PCA)")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
ax.legend()
plt.show()

from sklearn.naive_bayes import GaussianNB

# PCA (3 boyut)
X_pca_3d = pca.fit_transform(X)

# Naive Bayes modeli
nb = GaussianNB()
nb.fit(X_pca_3d, y)

# Sınıf merkezleri (ortalama)
class_means = nb.theta_

# 3D çizim
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(X_pca_3d[y == 0, 0], X_pca_3d[y == 0, 1], X_pca_3d[y == 0, 2],
           alpha=0.3, label="Sahte")
ax.scatter(X_pca_3d[y == 1, 0], X_pca_3d[y == 1, 1], X_pca_3d[y == 1, 2],
           alpha=0.3, label="Gerçek")

# Olasılık merkezleri
ax.scatter(class_means[:, 0], class_means[:, 1], class_means[:, 2],
           color="black", s=150, marker="X", label="Sınıf Merkezleri")

ax.set_title("Naive Bayes – Olasılık Merkezleri (PCA 3B)")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
ax.legend()
plt.show()
