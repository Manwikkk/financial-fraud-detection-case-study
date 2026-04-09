<![CDATA[<div align="center">

# 🔍 Financial Fraud Detection — Case Study

<img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/XGBoost-ML_Model-FF6600?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost">
<img src="https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
<img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
<img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=plotly&logoColor=white" alt="Matplotlib">
<img src="https://img.shields.io/badge/Seaborn-Visualization-4C72B0?style=for-the-badge" alt="Seaborn">

<br/>
<br/>

> **An end-to-end machine learning pipeline for detecting fraudulent financial transactions using the PaySim synthetic dataset (~6.36 million rows).**

<br/>

*Built as a case study for [Accredian](https://www.accredian.com/)*

---

</div>

<br/>

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Pipeline](#-project-pipeline)
- [Key Findings](#-key-findings)
- [Model Performance](#-model-performance)
- [Business Recommendations](#-business-recommendations)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [Tech Stack](#-tech-stack)

<br/>

---

## 🎯 Problem Statement

Financial fraud costs banks and fintech companies **billions every year**. The goal of this case study is to:

1. **Explore & understand** patterns in fraudulent transactions
2. **Build a machine learning model** that can reliably detect fraud (optimized for **high recall**)
3. **Suggest practical steps** a company can take to reduce fraud losses

<br/>

---

## 📊 Dataset

| Property | Details |
|:---|:---|
| **Name** | PaySim — Synthetic Financial Transaction Data |
| **Source** | [📥 Kaggle — Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| **Rows** | ~6,362,620 |
| **Columns** | 11 |
| **Simulation** | 30 days (744 hourly steps) |
| **Fraud Rate** | 0.13% (773:1 class imbalance) |

> ⚠️ **Note:** The dataset is not included in this repository due to its large size (~470 MB). Please download it from the Kaggle link above.

### Column Descriptions

| Column | Type | Description |
|:---|:---|:---|
| `step` | Integer | Time unit (1 step = 1 hour). 744 steps total (~30 days) |
| `type` | Categorical | Transaction type: `CASH-IN`, `CASH-OUT`, `DEBIT`, `PAYMENT`, `TRANSFER` |
| `amount` | Float | Transaction amount in local currency |
| `nameOrig` | String | Customer who initiated the transaction |
| `oldbalanceOrg` | Float | Sender's balance before the transaction |
| `newbalanceOrig` | Float | Sender's balance after the transaction |
| `nameDest` | String | Recipient (C = Customer, M = Merchant) |
| `oldbalanceDest` | Float | Recipient's balance before the transaction |
| `newbalanceDest` | Float | Recipient's balance after the transaction |
| `isFraud` | Binary | **Target** — 1 if fraudulent, 0 otherwise |
| `isFlaggedFraud` | Binary | Flagged by business rules (transfers > 200,000) |

Refer to the [`Data Dictionary.txt`](Data%20Dictionary.txt) file for full details.

<br/>

---

## 🛠️ Project Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    DATA LOADING                         │
│         Load CSV · Inspect Shape · Basic Stats          │
├─────────────────────────────────────────────────────────┤
│                 DATA UNDERSTANDING                      │
│    Column Types · Class Distribution · Null Checks      │
├─────────────────────────────────────────────────────────┤
│                   DATA CLEANING                         │
│  Missing Values · Duplicates · Merchant Flag Creation   │
├─────────────────────────────────────────────────────────┤
│             EXPLORATORY DATA ANALYSIS                   │
│  Transaction Types · Amount Distributions · Fraud       │
│  Patterns · Correlation Analysis · Time Patterns        │
├─────────────────────────────────────────────────────────┤
│               FEATURE ENGINEERING                       │
│  Balance Deltas · Error Flags · Transaction Ratios ·    │
│  Velocity Features · Temporal Features                  │
├─────────────────────────────────────────────────────────┤
│              MODEL BUILDING (XGBoost)                   │
│  Filter Fraud-Prone Types · Stratified Split ·          │
│  scale_pos_weight · Hyperparameter Tuning               │
├─────────────────────────────────────────────────────────┤
│                   EVALUATION                            │
│  Confusion Matrix · Classification Report ·             │
│  ROC-AUC · Feature Importance                           │
├─────────────────────────────────────────────────────────┤
│            BUSINESS RECOMMENDATIONS                     │
│  Actionable fraud prevention strategies                 │
└─────────────────────────────────────────────────────────┘
```

<br/>

---

## 🔑 Key Findings

<table>
  <tr>
    <td>💡</td>
    <td><strong>Fraud only occurs</strong> in <code>TRANSFER</code> and <code>CASH_OUT</code> transaction types</td>
  </tr>
  <tr>
    <td>💡</td>
    <td>Fraudulent transactions tend to <strong>completely drain</strong> the sender's account</td>
  </tr>
  <tr>
    <td>💡</td>
    <td>Massive <strong>class imbalance</strong> exists — only 0.13% of transactions are fraud (773:1 ratio)</td>
  </tr>
  <tr>
    <td>💡</td>
    <td>Balance discrepancies between expected and actual values are a <strong>strong fraud signal</strong></td>
  </tr>
  <tr>
    <td>💡</td>
    <td>The existing <code>isFlaggedFraud</code> system catches <strong>almost none</strong> of the actual fraud cases</td>
  </tr>
</table>

<br/>

---

## 📈 Model Performance

The final **XGBoost** model was trained on `TRANSFER` and `CASH_OUT` transactions with engineered features and class imbalance handling via `scale_pos_weight`.

| Metric | Score |
|:---|:---|
| **ROC-AUC** | ~0.99+ |
| **Recall (Fraud)** | ~96%+ |
| **Precision (Fraud)** | ~99%+ |
| **F1-Score (Fraud)** | ~97%+ |

> 🏆 The model achieves excellent recall, which is the most critical metric for fraud detection — **missing a fraud case is far more costly than a false alarm**.

### Top Features Driving the Model

1. `errorBalanceOrig` — Discrepancy in sender's balance
2. `errorBalanceDest` — Discrepancy in receiver's balance
3. `amount` — Transaction amount
4. `oldbalanceOrg` — Sender's initial balance
5. `newbalanceDest` — Receiver's new balance

<br/>

---

## 💼 Business Recommendations

Based on the analysis, here are actionable strategies for fraud prevention:

1. **🔒 Focus monitoring on TRANSFER & CASH_OUT** — These are the only channels where fraud occurs
2. **📊 Implement balance-delta checks** — Flag transactions where sender balance drops to zero unexpectedly
3. **⚡ Real-time scoring** — Deploy the XGBoost model for instant transaction risk scoring
4. **🚫 Replace the current flagging system** — `isFlaggedFraud` is ineffective; ML-based detection is far superior
5. **🔄 Velocity monitoring** — Track rapid sequences of large transfers from the same account
6. **📱 Multi-factor authentication** — Require additional verification for high-risk transactions

<br/>

---

## 📁 Repository Structure

```
📦 financial-fraud-detection-case-study
├── 📓 Fraud_Detection_Case_Study.ipynb   # Main analysis notebook (complete pipeline)
├── 📄 Data Dictionary.txt                # Column descriptions for the dataset
└── 📄 README.md                          # This file
```

> 📥 **Dataset:** Download `Fraud.csv` from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and place it in the root directory before running the notebook.

<br/>

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Manwikkk/financial-fraud-detection-case-study.git
cd financial-fraud-detection-case-study
```

### 2. Download the Dataset
Download from [Kaggle — PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) and place `Fraud.csv` in the project root.

### 3. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### 4. Run the Notebook
```bash
jupyter notebook Fraud_Detection_Case_Study.ipynb
```

<br/>

---

## 🧰 Tech Stack

| Tool | Purpose |
|:---|:---|
| **Python 3.x** | Core programming language |
| **Pandas** | Data manipulation & analysis |
| **NumPy** | Numerical computations |
| **Matplotlib** | Data visualization |
| **Seaborn** | Statistical plots |
| **Scikit-learn** | Model evaluation, preprocessing, train-test split |
| **XGBoost** | Gradient boosted classification model |
| **Jupyter Notebook** | Interactive development environment |

<br/>

---

<div align="center">

### ⭐ If you found this project helpful, please consider giving it a star!

<br/>

*Made with ❤️ by [Manwikkk](https://github.com/Manwikkk)*

</div>
]]>
