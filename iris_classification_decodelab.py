# DecodeLabs Project 2 - Data classification 
# Iris + KNN. Got 96.67% accuracy, 1 wrong out of 30.

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, classification_report,accuracy_score, f1_score)
import numpy as np
import matplotlib.pyplot as plt


dataset = load_iris()
X = dataset.data
y = dataset.target

print("Features:", dataset.feature_names)
print("Classes :", dataset.target_names)
print("Shape   :", X.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64, stratify=y)


# scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# training
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
predictions = knn.predict(X_test_scaled)


print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("F1 (macro):", round(f1_score(y_test, predictions, average='macro'), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n",classification_report(y_test, predictions, target_names=dataset.target_names))

# setosa is perfect, the only error is versicolor vs virginica . they overlap

# picking K with cross validation instead of just guessing 5
k_range = range(1, 31)
scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train_scaled, y_train, cv=5).mean()
for k in k_range]

plt.figure(figsize=(8, 4))
plt.plot(k_range, [1 - s for s in scores], marker='o')
plt.xlabel("K Value")
plt.ylabel("Cross-Validated Error Rate")
plt.title("Choosing K: The Elbow")
plt.grid(alpha=0.3)
plt.show()

best_k = k_range[int(np.argmax(scores))]
print("Best K:", best_k, "| CV accuracy:", round(max(scores), 4))


# checking if scaling even matters here. it didn't, same score both ways.
# all 4 features are already in similar cm ranges
knn_raw = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
print("\nKNN without scaling:", round(accuracy_score(y_test, knn_raw.predict(X_test)), 4))
print("KNN with scaling   :", round(accuracy_score(y_test, predictions), 4))


models = {
    "KNN (k=5)":          KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree":       DecisionTreeClassifier(random_state=64),
    "SVM (RBF)":           SVC(),
}

print("\n--- Algorithm Comparison ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"{name:22s} {acc:.4f}")
# all four tie at 0.9667 and make the same mistake, so the limit is the data
