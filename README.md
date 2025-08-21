#  Corporate Credit Rating Prediction (XGBoost + K-Means)
This project explores how financial ratios can be used to predict corporate credit ratings, inspired by the way agencies like S&amp;P or Fitch assign ratings to issuers of bonds.
Objective: Predict whether a company is Investment Grade or High Yield (Junk) using machine learning.

# Credit Ratings importance - Criteria/Regulation
- **Credit Ratings** = grades given to companies (AAA to D). Corporate ratings summarize a firm’s ability to repay debt.
- **Investment Grade** (BBB– and above): Safer, lower yields, attract pension funds & insurers.
- **High Yield** (“Junk”): Riskier, higher yields, often cyclical or leveraged companies.
- **Basel III**: Post-2008 regulations that require banks to hold more capital against riskier loans. Ergo, credit rating prediction directly impacts lending capacity.

# Datasets
| Dataset                           | Source                                                                                                 | Obs.  | Features                           | Notes                                         |
| --------------------------------- | ------------------------------------------------------------------------------------------------------ | ----- | ---------------------------------- | --------------------------------------------- |
| Corporate Credit Rating w/ Ratios | [Kaggle](https://www.kaggle.com/datasets/kirtandelwadia/corporate-credit-rating-with-financial-ratios) | 2,029 | \~30 ratios + multi-agency ratings | Liquidity, profitability, leverage, cash flow details |
| Corporate Credit Rating           | [Kaggle](https://www.kaggle.com/datasets/agewerc/corporate-credit-rating/data)                         | 7,805 | \~16 ratios + S\&P ratings         | 22-grade scale + IG vs HY                     |

# Modeling Pipeline - Methodology
1) Data ingestion & cleaning
   -> Clean missing values, normalize ratios, construct IG/HY binary flag.
2) Prediction models
   -> Logistic Regression for an interpretable baseline.
   -> XGBoost, provides a stronger predictive benchmark with SHAP feature importance.
3) Segmentation
   -> K-means clustering on leverage, profitability, and cash flow, as in to identify “borrower archetypes.”
4) Evaluation & Basel III Simulation
   -> A ROC/AUC analysis was used to measure predictive power across thresholds.
   -> To simulate capital allocation decisions:
   * If cutoff = 0.35 → reject 20% more HY, defaults drop, but portfolio shrinks.
   * If cutoff = 0.5 → accept more risk, higher yield but higher expected loss.

# Run This Notebook
- Clone the repo:
git clone https://github.com/SALanda/bonos-corpo-XGB-KMC
cd bonos-corpo-XGB-KMC
- Install dependencies:
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap
- Run in Jupyter/Colab:
jupyter notebook bonos-corpo-XGB-KMC.ipynb

# What to expect
- Which financial ratios drive corporate credit ratings.
- How logistic regression (transparent) compares to XGBoost (powerful).
- How clustering reveals risk segments beyond ratings.
- How Basel III constraints make rating predictions directly affect portfolio size and risk.

