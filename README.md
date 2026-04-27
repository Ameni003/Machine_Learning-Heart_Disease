# 🫀 Heart Disease UCI — Machine Learning Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-006400?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)


**Modèle de classification binaire pour prédire la présence de maladie cardiaque**  
à partir de données cliniques — UCI Heart Disease Dataset (920 patients)

[📓 Notebook](#-structure-du-notebook) • [📊 Résultats](#-résultats) • [🚀 Installation](#-installation) • [📁 Structure](#-structure-du-projet)

</div>

---

## 📋 Contexte

Les maladies cardiovasculaires représentent la **première cause de mortalité mondiale** (~17,9 millions de décès/an, OMS). Ce projet développe un pipeline ML complet pour **assister le diagnostic précoce** à partir de données cliniques.

### Objectifs
- 🔍 Explorer et comprendre les données médicales disponibles
- 📊 Identifier les facteurs les plus prédictifs de la maladie cardiaque
- ⚖️ Comparer 8 algorithmes de classification
- 🎯 Optimiser les performances via GridSearchCV
- 🏆 Sélectionner le meilleur modèle pour déploiement

---

## 📂 Structure du Projet

```
Machine_Learning-Heart_Disease/
│
├── 📋 cahier_de_charge.md            # Spécifications du projet
├── 📋 cahier_de_charge.pdf           # Spécifications du projet
├── 📓 heart_disease_analysis.ipynb   # Notebook principal (9 étapes)
├── 📊 heart_disease_uci.csv          # Dataset source (UCI)
└── 📄 README.md                      # Ce fichier
```

---

## 📊 Dataset


| Attribut | Valeur |
|----------|--------|
| Observations | **920 patients** |
| Variables | **16** (14 prédictives) |
| Centres | Cleveland · Budapest · Zurich · Long Beach |
| Variable cible | `num` → binarisée (0 = sain, 1 = malade) |

### Variables principales

| Variable | Description | Type |
|----------|-------------|------|
| `age` | Âge du patient | Numérique |
| `sex` | Sexe | Catégorielle |
| `cp` | Type de douleur thoracique | Catégorielle |
| `trestbps` | Pression artérielle au repos | Numérique |
| `chol` | Cholestérol sérique (mg/dl) | Numérique |
| `thalch` | Fréquence cardiaque max | Numérique |
| `oldpeak` | Dépression ST (exercice) | Numérique |
| `ca` | Nb de vaisseaux colorés | Numérique |
| `thal` | Thalassémie | Catégorielle |
| `num` | **Cible** — Diagnostic | Binaire |

> ⚠️ **Problèmes identifiés** : valeurs manquantes (`ca` ~65%, `slope` ~33%, `thal` ~30%), valeurs aberrantes (`chol=0`, `trestbps=0`), encodage des variables catégorielles.

---

## 📓 Structure du Notebook

Le notebook est organisé en **9 étapes** :

```
Étape 1 ── Imports & Chargement des données
Étape 2 ── Analyse Exploratoire (EDA)
           ├─ Statistiques descriptives
           ├─ Valeurs manquantes
           ├─ Distributions (num + catégorielles)
           ├─ Détection outliers (boxplots)
           ├─ Matrice de corrélation
           └─ Relations variables vs cible
Étape 3 ── Nettoyage & Prétraitement
           ├─ Imputation (médiane / mode)
           ├─ Correction valeurs aberrantes
           ├─ LabelEncoder
           ├─ Split 80/20 stratifié (random_state=42)
           └─ StandardScaler
Étape 4 ── Modèles Baseline (8 modèles)
Étape 5 ── Évaluation (confusion, ROC, F1...)
Étape 6 ── Backward Elimination (statsmodels OLS)
Étape 7 ── Fine-Tuning (GridSearchCV 5-folds)
Étape 8 ── Comparaison Finale Baseline vs Tuned
Étape 9 ── Conclusion & Recommandations
```

---

## 🤖 Modèles Comparés

| # | Modèle | Type |
|---|--------|------|
| 1 | Logistic Regression (L2) | Linéaire — Ridge |
| 2 | Logistic Regression (L1) | Linéaire — Lasso |
| 3 | Logistic Regression (ElasticNet) | Linéaire — L1+L2 |
| 4 | SVC | Kernel — hyperplan optimal |
| 5 | K-Nearest Neighbors | Instance-based |
| 6 | Decision Tree | Arbre de décision |
| 7 | Random Forest | Ensemble — Bagging |
| 8 | XGBoost | Ensemble — Boosting |

---

## 🏆 Résultats

> Résultats obtenus après GridSearchCV (5-folds, scoring=ROC AUC, random_state=42).

### 🥇 Meilleur Modèle — Random Forest 

| Métrique | Score |
|----------|-------|
| **ROC AUC** | **0.9107** |
| **Accuracy** | **0.8098** |
| **F1-Score** | **0.8325** |

**Meilleurs hyperparamètres trouvés :**
```python
RandomForestClassifier(
    max_depth         = 5,
    min_samples_split = 5,
    n_estimators      = 200,
    random_state      = 42
)
```

### 📊 Métriques d'évaluation utilisées
- **ROC AUC** — capacité de discrimination (métrique principale du projet)
- **Accuracy** — taux de bonnes prédictions global
- **Precision / Recall / F1-Score** — par classe (sain / malade)
- **Matrices de confusion** — analyse des erreurs FP / FN

---

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/ameni003/machine_learning-heart_disease.git
```

### 4. Lancer Jupyter

```bash
jupyter notebook heart_disease_analysis.ipynb
```

> ⚠️ Assure-toi que `heart_disease_uci.csv` est bien dans le **même dossier** que le notebook avant d'exécuter `Run All`.

---

## 📦 requirements.txt

```txt
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
statsmodels>=0.13.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

---

## 🔧 Pipeline ML

```
heart_disease_uci.csv
        │
        ▼
┌───────────────────────────────────────┐
│  EDA — Exploration & Visualisation    │
│  (distributions, outliers, corrélation│
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Preprocessing                        │
│  • chol=0 / trestbps=0  → médiane     │
│  • Imputation : médiane / mode        │
│  • LabelEncoder (7 variables)         │
│  • Train/Test split 80/20 stratifié   │
│  • StandardScaler                     │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Baseline — 8 modèles (défaut)        │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Backward Elimination (α = 0.05)      │
│  statsmodels OLS → variables p < 0.05 │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  GridSearchCV — Fine-Tuning           │
│  5-folds CV · scoring = ROC AUC       │
│  n_jobs = -1                          │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  🏆 Random Forest                     │
│     ROC AUC  = 0.9107                 │
│     Accuracy = 0.8098                 │
│     F1-Score = 0.8325                 │
│     max_depth=5, n_estimators=200     │
└───────────────────────────────────────┘
```

---

## 📈 Visualisations Produites

- 📊 Barplot des valeurs manquantes par variable
- 🥧 Pie chart distribution cible (sain / malade)
- 📈 Histogrammes + KDE avec lignes moyenne / médiane
- 📦 Boxplots détection d'outliers (IQR)
- 🔗 Heatmap matrice de corrélation triangulaire
- 📊 Stacked barplots variables catégorielles vs cible
- 🔢 Grille 2×4 matrices de confusion (baseline + tuned)
- 📈 Courbes ROC superposées (tous modèles)
- 📊 Barplots comparatifs Accuracy / F1 / AUC (Baseline vs Tuned)
- 📈 Courbes ROC côte à côte (Baseline vs Tuned)

---

## ⚠️ Limites

| Limite | Description |
|--------|-------------|
| Taille dataset | 920 observations — petit pour le deep learning |
| Multi-centres | 4 sources → variabilité inter-hospitalière |
| Binarisation | Simplification de la variable 0–4 en 0/1 (perte d'information sur la sévérité) |
| Valeurs manquantes | Jusqu'à 65% pour `ca` — imputation par médiane introduit du bruit |

---

## 🚀 Perspectives

- **Déploiement** — API REST avec Flask / FastAPI + Docker
- **Interprétabilité** — SHAP / LIME pour expliquer les prédictions aux médecins
- **Deep Learning** — MLP, TabNet sur dataset enrichi
- **Collecte de données** — Plus de patients pour améliorer la robustesse
- **Validation clinique** — Évaluation prospective en milieu hospitalier réel

---



## 👤 Réalisé par 

BOUAZIZ Ameni 
<br>
TRABELSI Asma
<br>


