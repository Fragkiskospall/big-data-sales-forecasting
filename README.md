# Big Data Sales Forecasting
### End-to-End Machine Learning Pipeline — Favorita Grocery Dataset

Πρόβλεψη κλάσης πωλήσεων (Low / Medium / High) σε 54 καταστήματα του Ισημερινού (2013–2017).

---

## Δομή Project

```
big-data-sales-forecasting/
├── data/
│   ├── raw/                       # Αρχικά CSV (train, stores, oil, holidays, transactions)
│   └── processed/                 # Επεξεργασμένα Parquet (train/val/test)
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Φάση 1a: Data Understanding
│   ├── 02_preprocessing.ipynb    # Φάση 1b: Preprocessing Pipeline
│   ├── 03_eda.ipynb              # Φάση 2: Exploratory Data Analysis
│   ├── 04_spark.ipynb            # Φάση 3: Big Data Processing (PySpark)
│   ├── 05_naive_bayes_svm.ipynb  # Φάση 3: Naive Bayes + SVM
│   ├── 06_association_rules.ipynb # Φάση 3: Association Rules (Apriori)
│   ├── 07_neural_networks.ipynb  # Φάση 3: Neural Networks (MLP)
│   └── 08_evaluation.ipynb       # Φάση 4: Comparative Evaluation
├── requirements.txt
└── README.md
```

---

## Dataset

**Πηγή**: [Kaggle — Store Sales Time Series Forecasting (Favorita)](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

| Αρχείο | Γραμμές | Περιγραφή |
|--------|---------|-----------|
| `train.csv` | 3,000,888 | Ημερήσιες πωλήσεις ανά κατάστημα & κατηγορία |
| `stores.csv` | 54 | Τύπος, πόλη, πολιτεία, cluster |
| `oil.csv` | 1,218 | Ημερήσια τιμή πετρελαίου (WTI) |
| `holidays_events.csv` | 350 | Εθνικές / τοπικές αργίες |
| `transactions.csv` | 83,488 | Ημερήσιες συναλλαγές ανά κατάστημα |

**Κατηγορίες προϊόντων (families)**: 31 (GROCERY I, BEVERAGES, PRODUCE, MEATS, …)

---

## Φάσεις Εργασίας

### Φάση 1: Preprocessing (`02_preprocessing.ipynb`)
- Joins: train + stores + oil + transactions
- Oil price forward/backward fill (525 missing values)
- Holiday flags: national / regional / local
- Date features: year, month, day_of_week, is_weekend, is_month_start, …
- Φιλτράρισμα structural zeros (−16.8% γραμμές)
- **Target**: `sales_class` (Low ≤ 4, Medium ≤ 150, High > 150) — θρεσχόλδια από train set μόνο
- One-hot encoding (family × 31, type × 5, state × 16)
- Memory optimization: 1,437 MB → **286 MB**
- Χρονικός split: Train (2013–2016) / Val (Jan–May 2017) / Test (Jun–Aug 2017)

| Set | Rows | Period |
|-----|------|--------|
| Train | 2,160,731 | 2013-01-01 → 2016-12-31 |
| Val | 223,933 | 2017-01-01 → 2017-05-31 |
| Test | 112,708 | 2017-06-01 → 2017-08-15 |

---

### Φάση 2: EDA (`03_eda.ipynb`)
- Κατανομή πωλήσεων (heavy right-skew, log1p transform)
- Top families: GROCERY I > BEVERAGES > PRODUCE
- Εποχικότητα: peak Δεκέμβριο, Σαββατοκύριακα υψηλότερες πωλήσεις
- Αργίες: +~15% μέσες πωλήσεις σε εθνικές αργίες
- Αρνητική συσχέτιση τιμής πετρελαίου — πωλήσεων
- Γεωγραφία: Pichincha (Quito) κυριαρχεί

---

### Φάση 3: Αλγόριθμοι

#### Big Data Processing (`04_spark.ipynb`)
- PySpark DataFrame API
- SparkSQL aggregations (top families, monthly trend, store type analysis)
- Window functions για oil price forward-fill
- Distributed preprocessing pipeline (joins, date features, classification)
- Partitioned Parquet output
- Pandas vs Spark comparison

#### Naive Bayes & SVM (`05_naive_bayes_svm.ipynb`)
- **Gaussian NB**: Full 2.1M training set, ~1s training
- **Bernoulli NB**: Binary features only (family/type/state/holiday flags)
- **LinearSVC**: 200K sample + StandardScaler, C=1.0

#### Association Rules (`06_association_rules.ipynb`)
- Library: `mlxtend` (Apriori + FP-Growth)
- Transaction: (store_nbr, date) → list of families with sales > 0
- Dataset: 2016 (full year, all stores)
- min_support=0.30, min_confidence=0.70
- Ευρήματα: DAIRY↔EGGS, MEATS↔POULTRY ισχυρές συσχετίσεις

#### Neural Networks (`07_neural_networks.ipynb`)
- **MLPClassifier** (scikit-learn), 3 αρχιτεκτονικές:
  - Shallow: `(64,)` — 1 hidden layer
  - Medium: `(128, 64)` — 2 hidden layers
  - Deep: `(256, 128, 64)` — 3 hidden layers
- solver=adam, activation=relu, early_stopping=True, batch_size=1024
- 500K training sample + StandardScaler

---

### Φάση 4: Evaluation (`08_evaluation.ipynb`)
- Σύγκριση όλων μοντέλων: Accuracy, Macro F1, Precision, Recall
- Confusion matrices (6 μοντέλα ταυτόχρονα)
- ROC curves (One-vs-Rest) με AUC
- 5-fold Cross-Validation (100K sample)
- Feature importance (HistGradientBoosting / Random Forest)

#### Αποτελέσματα (300K training sample)

| Rank | Μοντέλο | Val Macro F1 |
|------|---------|-------------|
| 🥇 1 | **HistGradientBoosting** | καλύτερο |
| 🥈 2 | Random Forest | ≈ HGB |
| 🥉 3 | MLP Neural Network | καλό |
| 4 | LinearSVC | σταθερό |
| 5 | Gaussian NB | γρήγορο |
| 6 | Bernoulli NB | baseline |

---

## Εγκατάσταση & Εκτέλεση

```bash
# Clone / navigate
cd big-data-sales-forecasting

# Activate venv
source venv/bin/activate

# Εκκίνηση Jupyter
jupyter lab

# Εκτέλεση notebooks με σειρά:
# 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

### Εξωτερικές βιβλιοθήκες (auto-install μέσα στα notebooks)
```bash
pip install mlxtend   # Association Rules (notebook 06)
pip install pyspark   # Big Data (notebook 04)
```

---

## Τεχνολογίες

| Κατηγορία | Εργαλεία |
|-----------|---------|
| Language | Python 3.11 |
| Data | pandas 3.0, numpy 2.4, pyarrow 24 |
| ML | scikit-learn 1.8, mlxtend |
| Visualization | matplotlib 3.10, seaborn 0.13 |
| Big Data | Apache Spark (PySpark) |
| Environment | Jupyter Lab, venv |

---

## Ρόλοι Ομάδας

| Ρόλος | Notebooks | Κεφάλαια Report |
|-------|-----------|-----------------|
| Data Engineer & EDA Specialist | 01, 02, 03, 04 | Dataset Description, Preprocessing, EDA, Big Data |
| Classical ML Specialist | 05, 06 | Naive Bayes, SVM, Association Rules |
| Deep Learning & Evaluation Lead | 07, 08 | Neural Networks, Comparative Evaluation, Conclusions |
