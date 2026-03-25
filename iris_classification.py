# ============================================================
# Iris Flower Classification — CodeAlpha Task 1
# ============================================================

# ── Imports ──────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

import warnings
warnings.filterwarnings("ignore")

# ── 1. Load Dataset ───────────────────────────────────────────
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("=" * 55)
print("           IRIS FLOWER CLASSIFICATION")
print("=" * 55)
print("\n📊 Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass Distribution:")
print(df['species'].value_counts())
print("\nBasic Statistics:")
print(df.describe().round(2))

# ── 2. Exploratory Data Analysis (EDA) ───────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Iris Dataset — Feature Distributions by Species", fontsize=14, fontweight='bold')

features = iris.feature_names
colors   = ['#E74C3C', '#2ECC71', '#3498DB']

for ax, feat in zip(axes.flatten(), features):
    for cls, clr in zip(iris.target_names, colors):
        ax.hist(df[df['species'] == cls][feat], bins=15,
                alpha=0.6, label=cls, color=clr, edgecolor='white')
    ax.set_title(feat)
    ax.set_xlabel("Value (cm)")
    ax.set_ylabel("Frequency")
    ax.legend()

plt.tight_layout()
plt.savefig("iris_feature_distributions.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Saved: iris_feature_distributions.png")

# Pair Plot
sns.set(style="ticks", font_scale=1.1)
pairplot = sns.pairplot(df, hue='species', palette={'setosa':'#E74C3C',
                                                     'versicolor':'#2ECC71',
                                                     'virginica':'#3498DB'},
                        diag_kind='kde', plot_kws={'alpha': 0.6})
pairplot.fig.suptitle("Pair Plot — Iris Dataset", y=1.02, fontsize=14, fontweight='bold')
plt.savefig("iris_pairplot.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: iris_pairplot.png")

# Correlation Heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df.drop('species', axis=1).corr(), annot=True, fmt=".2f",
            cmap='coolwarm', linewidths=0.5, square=True)
plt.title("Feature Correlation Heatmap", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("iris_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: iris_correlation_heatmap.png")

# ── 3. Data Preprocessing ─────────────────────────────────────
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\n🔀 Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── 4. Model Training & Comparison ───────────────────────────
models = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

print("\n" + "=" * 55)
print("           MODEL COMPARISON (5-Fold CV)")
print("=" * 55)

results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = cv_scores
    print(f"\n{name}")
    print(f"  CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 5. Best Model — Final Evaluation ─────────────────────────
best_model = models["Random Forest"]
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\n" + "=" * 55)
print("     FINAL EVALUATION — Random Forest")
print("=" * 55)
print(f"\nTest Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ── 6. Confusion Matrix ───────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title("Confusion Matrix — Random Forest", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("iris_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: iris_confusion_matrix.png")

# ── 7. Feature Importance ─────────────────────────────────────
feat_imp = pd.Series(best_model.feature_importances_, index=iris.feature_names).sort_values()
plt.figure(figsize=(7, 4))
feat_imp.plot(kind='barh', color=['#3498DB', '#2ECC71', '#F39C12', '#E74C3C'])
plt.title("Feature Importance — Random Forest", fontsize=13, fontweight='bold')
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("iris_feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: iris_feature_importance.png")

# ── 8. Model Comparison Bar Chart ────────────────────────────
mean_scores = {name: scores.mean() for name, scores in results.items()}
plt.figure(figsize=(8, 5))
bars = plt.bar(mean_scores.keys(), mean_scores.values(),
               color=['#3498DB', '#E74C3C', '#2ECC71'], edgecolor='white', width=0.5)
for bar, val in zip(bars, mean_scores.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.02,
             f"{val:.4f}", ha='center', va='top', color='white', fontweight='bold')
plt.ylim(0.9, 1.01)
plt.title("Model Accuracy Comparison (CV)", fontsize=13, fontweight='bold')
plt.ylabel("Mean CV Accuracy")
plt.tight_layout()
plt.savefig("iris_model_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: iris_model_comparison.png")

print("\n🎉 Task 1 Complete! All outputs saved.")
