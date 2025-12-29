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
        ("model", SVC(kernel="rbf", random_state=42))
    ]),
    "kNN": Pipeline([
        ("model", KNeighborsClassifier(n_neighbors=5))
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
