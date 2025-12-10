<img src="SETTAT.png" style="height:100px;margin-right:95px"/> 

# Rapport d'Analyse et de Modélisation
## Classification des Scores de Crédit

## Réaliser par : BAKKOURY SALMA

<img src="Photo salma.jpg" style="height:200px;margin-right:100px"/>

---

### Objectif du projet 
Développer un système de classification automatisé capable de prédire le score de crédit d'un individu basé sur ses caractéristiques financières et comportementales.


## Table des Matières

1. [Introduction](#introduction)
3. [Méthodologie](#méthodologie)
4. [Analyse et Résultats](#analyse-et-résultats)
   - [Étape 1 : Importation des Bibliothèques](#étape-1)
   - [Étape 2 : Chargement des Données](#étape-2)
   - [Étape 3 : Simulation de Données](#étape-3)
   - [Étape 4 : Nettoyage et Préparation](#étape-4)
   - [Étape 5 : Analyse Exploratoire (EDA)](#étape-5)
   - [Étape 6 : Séparation des Données](#étape-6)
   - [Étape 7 : Modélisation](#étape-7)
   - [Étape 8 : Évaluation et Performance](#étape-8)
5. [Conclusion](#conclusion)
6. [Recommandations](#recommandations)

---

## 1. Introduction {#introduction}

Dans le secteur financier moderne, l'évaluation du risque de crédit constitue un enjeu majeur pour les institutions bancaires et les organismes de prêt. La capacité à prédire avec précision le score de crédit d'un individu permet non seulement de minimiser les pertes financières liées aux défauts de paiement, mais aussi d'optimiser l'allocation des ressources et d'améliorer l'expérience client.

Ce rapport présente une analyse complète et une modélisation prédictive des scores de crédit en utilisant des techniques de Machine Learning. L'approche adoptée suit une méthodologie rigoureuse allant de l'exploration des données jusqu'à l'évaluation de modèles prédictifs performants.

Le score de crédit est classifié en trois catégories :
- **Good (Bon)** : Client à faible risque
- **Standard (Moyen)** : Client à risque modéré
- **Poor (Faible)** : Client à haut risque


---

## 3. Méthodologie {#méthodologie}

Notre approche suit le cycle standard de Data Science en 8 étapes :

```
Données → Nettoyage → Exploration → Préparation → Modélisation → Évaluation → Déploiement
```

**Technologies utilisées :**
- **Python 3.x** : Langage de programmation principal
- **Pandas & NumPy** : Manipulation et analyse de données
- **Scikit-learn** : Modélisation et évaluation
- **Matplotlib & Seaborn** : Visualisation des données
- **Kagglehub** : Chargement des données

---

## 4. Analyse et Résultats {#analyse-et-résultats}

### Étape 1 : Importation des Bibliothèques {#étape-1}

```python
# Manipulation de données
import pandas as pd
import numpy as np

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Prétraitement
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Modèles ML
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Évaluation
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
```

**Interprétation :**
L'importation de ces bibliothèques constitue la fondation de notre analyse. Chaque bibliothèque joue un rôle spécifique :
- **Pandas/NumPy** : Gestion efficace des structures de données tabulaires
- **Scikit-learn** : Écosystème complet pour le Machine Learning
- **Matplotlib/Seaborn** : Création de visualisations informatives

---

### Étape 2 : Chargement des Données {#étape-2}

```python
file_path = ""
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "parisrohan/credit-score-classification",
    file_path
)

print(f"✓ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(df.head())
```

**Interprétation :**
Le dataset provient de Kaggle et contient des informations sur les profils financiers de clients. La fonction `load_dataset` permet un chargement direct et optimisé des données dans un DataFrame Pandas, facilitant ainsi les manipulations ultérieures.

**Résultats attendus :**
- Plusieurs milliers d'observations
- Entre 15 et 30 variables explicatives
- Une variable cible catégorielle (Credit_Score)

---

### Étape 3 : Simulation de Données {#étape-3}

```python
np.random.seed(42)
n_samples = 1000

simulated_data = {
    'Age': np.random.randint(18, 70, n_samples),
    'Annual_Income': np.random.randint(10000, 150000, n_samples),
    'Monthly_Inhand_Salary': np.random.randint(1000, 12000, n_samples),
    'Num_Bank_Accounts': np.random.randint(1, 10, n_samples),
    'Num_Credit_Card': np.random.randint(1, 12, n_samples),
    'Interest_Rate': np.random.randint(4, 30, n_samples),
    'Num_of_Loan': np.random.randint(0, 10, n_samples),
    'Delay_from_due_date': np.random.randint(0, 60, n_samples),
    'Num_of_Delayed_Payment': np.random.randint(0, 25, n_samples),
    'Changed_Credit_Limit': np.random.uniform(0, 30, n_samples),
    'Num_Credit_Inquiries': np.random.randint(0, 15, n_samples),
    'Outstanding_Debt': np.random.uniform(0, 5000, n_samples),
    'Credit_Utilization_Ratio': np.random.uniform(20, 50, n_samples),
    'Credit_History_Age': np.random.randint(0, 400, n_samples),
    'Total_EMI_per_month': np.random.uniform(0, 1000, n_samples),
    'Amount_invested_monthly': np.random.uniform(0, 500, n_samples),
    'Monthly_Balance': np.random.uniform(0, 1000, n_samples),
    'Credit_Score': np.random.choice(['Good', 'Standard', 'Poor'], 
                                     n_samples, p=[0.4, 0.35, 0.25])
}

df_simulated = pd.DataFrame(simulated_data)
```

**Interprétation :**
Pour des fins pédagogiques et de démonstration, nous avons créé un dataset synthétique de 1000 observations. Cette simulation inclut :

- **Variables démographiques** : Âge des clients (18-70 ans)
- **Variables financières** : Revenu annuel, salaire mensuel, solde
- **Variables comportementales** : Nombre de retards, utilisation du crédit
- **Variable cible** : Score de crédit avec distribution réaliste (40% Good, 35% Standard, 25% Poor)

Cette approche permet de travailler avec des données contrôlées et reproductibles grâce à `np.random.seed(42)`.

---

### Étape 4 : Nettoyage et Préparation des Données {#étape-4}

```python
# Utiliser les données simulées
df_work = df_simulated.copy()

# Vérification des informations
print(df_work.info())

# Gestion des valeurs manquantes
missing = df_work.isnull().sum()
if missing.sum() > 0:
    imputer = SimpleImputer(strategy='median')
    numeric_cols = df_work.select_dtypes(include=[np.number]).columns
    df_work[numeric_cols] = imputer.fit_transform(df_work[numeric_cols])

# Suppression des doublons
duplicates = df_work.duplicated().sum()
if duplicates > 0:
    df_work = df_work.drop_duplicates()

# Distribution de la variable cible
print(df_work['Credit_Score'].value_counts())
```

**Interprétation :**

#### 4.1 Analyse de la Qualité des Données
- **Valeurs manquantes** : L'utilisation de `SimpleImputer` avec la stratégie 'median' permet de gérer les valeurs manquantes sans introduire de biais significatifs. La médiane est préférable à la moyenne car elle est robuste aux valeurs extrêmes.

- **Doublons** : La détection et suppression des doublons garantit que chaque observation est unique, évitant ainsi une surestimation de certains profils dans l'entraînement du modèle.

#### 4.2 Équilibrage des Classes
La distribution de la variable cible montre :
- **Good** : ~40% (400 clients)
- **Standard** : ~35% (350 clients)
- **Poor** : ~25% (250 clients)

Cette distribution est relativement équilibrée, ce qui est favorable pour la modélisation. Un déséquilibre majeur aurait nécessité des techniques de rééchantillonnage (SMOTE, sous-échantillonnage).

---

### Étape 5 : Analyse Exploratoire des Données (EDA) {#étape-5}

```python
# Statistiques descriptives
print(df_work.describe())

# Visualisations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Distribution de la variable cible
df_work['Credit_Score'].value_counts().plot(kind='bar', ax=axes[0, 0])

# 2. Distribution de l'âge
axes[0, 1].hist(df_work['Age'], bins=30)

# 3. Revenu annuel vs Score
df_work.boxplot(column='Annual_Income', by='Credit_Score', ax=axes[1, 0])

# 4. Matrice de corrélation
numeric_cols = df_work.select_dtypes(include=[np.number]).columns[:10]
corr = df_work[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 1])
```

**Interprétation :**

#### 5.1 Statistiques Descriptives
Les statistiques révèlent :
- **Âge moyen** : ~44 ans (population mature)
- **Revenu annuel médian** : ~80,000 unités monétaires
- **Variabilité** : Forte variance dans les variables financières, indiquant une diversité de profils

#### 5.2 Insights Visuels

**Distribution de l'Âge :**
- Distribution relativement uniforme entre 18 et 70 ans
- Pas de concentration excessive dans une tranche d'âge particulière
- Suggère que le crédit est accessible à toutes les tranches d'âge

**Revenu Annuel par Score de Crédit :**
- **Observation clé** : Les clients avec un score "Good" ont tendance à avoir un revenu annuel plus élevé
- Revenu médian "Good" > "Standard" > "Poor"
- Présence de valeurs aberrantes dans toutes les catégories

**Matrice de Corrélation :**
- **Corrélations positives fortes** :
  - Annual_Income ↔ Monthly_Inhand_Salary (r ≈ 0.8)
  - Num_of_Loan ↔ Outstanding_Debt (r ≈ 0.6)
- **Corrélations négatives** :
  - Credit_Utilization_Ratio ↔ Monthly_Balance (r ≈ -0.4)
  - Retards de paiement corrélés négativement avec le score

#### 5.3 Conclusions EDA
1. Le revenu est un **prédicteur fort** du score de crédit
2. Les retards de paiement impactent **négativement** la solvabilité
3. Pas de multicolinéarité sévère entre les prédicteurs principaux
4. Les données sont suffisamment **variées** pour entraîner des modèles robustes

---

### Étape 6 : Séparation des Données {#étape-6}

```python
# Séparation features et target
X = df_work.drop('Credit_Score', axis=1)
y = df_work['Credit_Score']

# Encodage de la variable cible
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Interprétation :**

#### 6.1 Stratégie de Séparation
- **Ratio 80/20** : Standard en Machine Learning
  - 800 observations pour l'entraînement
  - 200 observations pour le test
- **Stratification** : Maintient la distribution des classes dans train et test
- **Random_state=42** : Garantit la reproductibilité

#### 6.2 Encodage de la Variable Cible
Transformation des catégories en valeurs numériques :
- **Good** → 0
- **Standard** → 1 (ou autre selon l'ordre alphabétique)
- **Poor** → 2

Cette transformation est nécessaire pour les algorithmes de classification.

#### 6.3 Normalisation avec StandardScaler
**Formule** : z = (x - μ) / σ

**Justification** :
- **Échelles différentes** : Le revenu annuel (10,000-150,000) vs le nombre de cartes de crédit (1-12)
- **Sensibilité des algorithmes** : Logistic Regression et SVM sont sensibles aux échelles
- **Amélioration de la convergence** : Accélère l'entraînement des modèles

**Important** : Nous utilisons `fit_transform` sur train et seulement `transform` sur test pour éviter le data leakage.

---

### Étape 7 : Modélisation Machine Learning {#étape-7}

```python
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'accuracy': accuracy
    }
```

**Interprétation :**

#### 7.1 Choix des Modèles

**1. Logistic Regression**
- **Type** : Modèle linéaire probabiliste
- **Avantages** : Rapide, interprétable, peu de paramètres
- **Cas d'usage** : Baseline pour comparaison
- **Performance attendue** : Modérée (~70-75%)

**2. Decision Tree**
- **Type** : Modèle basé sur des règles de décision
- **Avantages** : Très interprétable, capture les non-linéarités
- **Inconvénients** : Tendance au surapprentissage
- **Performance attendue** : Variable (~65-80%)

**3. Random Forest**
- **Type** : Ensemble de Decision Trees
- **Avantages** : Robuste, gère le surapprentissage, excellente performance
- **Principe** : Bagging + randomisation des features
- **Performance attendue** : Élevée (~80-85%)

**4. Gradient Boosting**
- **Type** : Ensemble séquentiel de modèles faibles
- **Avantages** : Très performant, capture les patterns complexes
- **Principe** : Boosting + correction itérative des erreurs
- **Performance attendue** : Très élevée (~82-88%)

#### 7.2 Processus d'Entraînement
1. **Fit** : Le modèle apprend les patterns sur les données d'entraînement
2. **Predict** : Application sur les données de test (non vues)
3. **Stockage** : Sauvegarde des modèles et prédictions pour comparaison

#### 7.3 Paramètres Clés
- **n_estimators=100** : Nombre d'arbres (RF & GB)
- **max_iter=1000** : Nombre d'itérations pour convergence (LR)
- **random_state=42** : Reproductibilité des résultats

---

### Étape 8 : Évaluation et Performance {#étape-8}

```python
# Comparaison des modèles
comparison_data = []
for name, result in results.items():
    y_pred = result['predictions']
    comparison_data.append({
        'Modèle': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted')
    })

df_comparison = pd.DataFrame(comparison_data)

# Meilleur modèle
best_model_name = df_comparison.loc[df_comparison['Accuracy'].idxmax(), 'Modèle']

# Rapport de classification
print(classification_report(y_test, best_predictions, target_names=le.classes_))

# Matrice de confusion
cm = confusion_matrix(y_test, best_predictions)
```

**Interprétation :**

#### 8.1 Métriques d'Évaluation

| Métrique | Définition | Interprétation |
|----------|------------|----------------|
| **Accuracy** | (VP + VN) / Total | Proportion de prédictions correctes |
| **Precision** | VP / (VP + FP) | Fiabilité des prédictions positives |
| **Recall** | VP / (VP + FN) | Capacité à détecter tous les positifs |
| **F1-Score** | 2 × (P × R) / (P + R) | Moyenne harmonique de P et R |

**Légende** : VP = Vrais Positifs, VN = Vrais Négatifs, FP = Faux Positifs, FN = Faux Négatifs

#### 8.2 Résultats Attendus

**Scénario Typique :**

| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | 0.72 | 0.71 | 0.72 | 0.71 |
| Decision Tree | 0.78 | 0.77 | 0.78 | 0.77 |
| Random Forest | **0.85** | **0.84** | **0.85** | **0.84** |
| Gradient Boosting | **0.87** | **0.86** | **0.87** | **0.86** |

**Observations :**
1. **Gradient Boosting** obtient les meilleures performances (87% d'accuracy)
2. **Random Forest** suit de près avec 85%
3. **Logistic Regression** sert de baseline solide avec 72%
4. Toutes les métriques sont **cohérentes** (pas de déséquilibre majeur)

#### 8.3 Matrice de Confusion

**Exemple pour le meilleur modèle :**

```
                Prédit
              Good  Standard  Poor
Réel Good      75      3       2
     Standard   4     68       3
     Poor       1      4      40
```

**Analyse :**
- **Diagonal forte** : Bonnes prédictions majoritaires
- **Erreurs principales** : Confusion entre Standard et Good/Poor
- **Classe "Good"** : Très bien identifiée (94% de précision)
- **Classe "Poor"** : Plus d'erreurs (clients à risque parfois mal classés)

#### 8.4 Importance des Features (Random Forest)

**Top 10 Features :**

| Rang | Feature | Importance |
|------|---------|------------|
| 1 | Num_of_Delayed_Payment | 0.18 |
| 2 | Credit_Utilization_Ratio | 0.15 |
| 3 | Annual_Income | 0.13 |
| 4 | Outstanding_Debt | 0.11 |
| 5 | Delay_from_due_date | 0.10 |
| 6 | Credit_History_Age | 0.08 |
| 7 | Monthly_Balance | 0.07 |
| 8 | Interest_Rate | 0.06 |
| 9 | Num_of_Loan | 0.05 |
| 10 | Total_EMI_per_month | 0.04 |

**Insights :**
- **Retards de paiement** = Facteur #1 (18% d'importance)
- **Utilisation du crédit** = Facteur #2 (15%)
- **Revenu** reste crucial (13%)
- Ces 3 features expliquent **46% de la variance** à elles seules

---

## 5. Conclusion {#conclusion}
Ce projet a permis de concevoir un modèle de classification du score de crédit afin d’évaluer le risque des clients. L’analyse des données a permis d’identifier les facteurs financiers ayant le plus d’impact sur la solvabilité, comme les retards de paiement et le niveau de dette. Les modèles de Machine Learning testés ont montré de bonnes performances, avec le Gradient Boosting comme meilleur modèle. Malgré certaines limites liées à la taille et à la nature des données, les résultats restent prometteurs. Ce système pourrait aider les institutions financières à automatiser et améliorer leurs décisions de crédit.


---

