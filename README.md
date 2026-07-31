# Iris Flower Classification

A beginner machine learning project that classifies iris flowers into one of three species based on flower measurements. This project introduces the core ML workflow: loading data, splitting it, training a classifier, and evaluating results.

## Project Goal

Predict the species of an iris flower — **Setosa**, **Versicolor**, or **Virginica** — using four numeric measurements of the flower.

## Dataset

The Iris dataset is a classic, small, clean dataset built directly into scikit-learn (`sklearn.datasets.load_iris`).

| Property | Value |
|---|---|
| Samples | 150 |
| Features | 4 (all numeric) |
| Classes | 3 (50 samples each — balanced) |
| Missing values | None |

**Features:**
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

**Target classes:**
- 0 = Setosa
- 1 = Versicolor
- 2 = Virginica

## Requirements

```
python >= 3.8
scikit-learn
pandas
numpy
matplotlib
```

Install with:
```bash
pip install scikit-learn pandas numpy matplotlib
```

## Workflow

### 1. Load the Data
```python
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data      # features
y = iris.target    # labels
```

### 2. Train/Test Split
Split the data so the model is evaluated on unseen samples, not the ones it trained on.
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
- `test_size=0.2` — 80% train, 20% test
- `stratify=y` — keeps class balance equal in both splits
- `random_state=42` — makes the split reproducible

### 3. Train a Classifier

**K-Nearest Neighbors (KNN)**
Classifies a point by majority vote among its `k` closest neighbors in feature space. No real "training" step — it stores the data and computes distances at prediction time.
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```

**Decision Tree**
Builds a sequence of if/else questions on feature values (e.g., "petal length < 2.5?") that split the data to maximize class purity at each step.
```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
```

### 4. Evaluate
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

preds = knn.predict(X_test)
print(accuracy_score(y_test, preds))
print(classification_report(y_test, preds, target_names=iris.target_names))
print(confusion_matrix(y_test, preds))
```

## Expected Results

Both KNN and Decision Tree typically reach **90–100% accuracy** on this dataset. Iris is considered an "easy" dataset because petal length and petal width alone almost perfectly separate the three species.

## Key Concepts Covered

- Loading a dataset into features (`X`) and labels (`y`)
- Why train/test splitting matters (avoiding evaluation on data the model already saw)
- Stratified sampling to preserve class balance
- Distance-based classification (KNN)
- Rule-based classification (Decision Tree)
- Accuracy, precision/recall, and confusion matrices as evaluation tools

## Extensions to Try

- Plot petal length vs. petal width, colored by species, to visually see class separation
- Sweep `n_neighbors` from 1–20 and plot accuracy vs. k to see the underfit/overfit tradeoff
- Sweep `max_depth` on the Decision Tree (1, 2, 3, `None`) and observe overfitting as depth grows
- Apply `StandardScaler` before KNN (distance-based models are sensitive to feature scale; trees are not)
- Replace the single train/test split with `cross_val_score` for a more robust accuracy estimate

## Project Structure (suggested)
```
iris-classification/
├── README.md
├── iris_classification.py   # or .ipynb
└── requirements.txt
```
