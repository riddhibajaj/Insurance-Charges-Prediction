# Insurance Charges Prediction
This project explores the factors impacting **medical insurance charges** and builds a predictive model using regression techniques.

The final output is a **Streamlit web app** where users can input their details (age, sex, BMI, smoker status, children, region) and get a predicted insurance cost.

The goal was to demonstrate end-to-end data science skills following the **CRISP-DM framework**, from data understanding and data preparation to feature engineering, modeling, evaluation, and deployment.

## Dataset
* Source: `insurance.csv` (Kaggle Insurance Dataset)
* 1,338 rows, 6 features, 1 target (charges)

### Data Dictionary
| Column     | Description                                                 |
| ---------- | ----------------------------------------------------------- |
| `age`      | Age of the customer (18–64)                                 |
| `sex`      | Biological sex (`male`, `female`)                           |
| `bmi`      | Body Mass Index                                             |
| `children` | Number of children covered by plan                          |
| `smoker`   | Smoker status (`yes`, `no`)                                 |
| `region`   | Region (`northeast`, `northwest`, `southeast`, `southwest`) |
| `charges`  | Annual insurance charges (target)                           |

## Key Insights
* **Smoker status** is the strongest predictor. Smokers pay several times more.
* **Age** and **BMI** also drive higher charges.
* **Children**, **sex**, **region** have weak impact.
* Target variable (`charges`) is right-skewed → applied **log transformation**.

## Data Preparation
* Handled duplicates (minimal).
* No missing values.
* Feature engineering:
  * Binary encoding (`sex`, `smoker`).
  * One-hot encoding (`region`).
  * Log transform on `charges`.
  * Standard scaling (`age`, `bmi`).
* Outliers: minor in BMI → handled by clipping at 99th percentile.

## Modeling
* Tried **Linear Regression**, **Ridge**, **Lasso**, **Decision Tree**, **Random Forest**, **Gradient Boosting**, **SVR**, **KNN**, **Poisson Regressor**.
* Evaluated with **R²** and **RMSE** on train/validation/test splits.
* Feature importance measured with **Permutation Feature Importance (PFI)**.

#### Top Features (PFI)
1. Smoker status
2. Age
3. BMI
4. Children (weak)
5. Sex/Region (very weak)

## Streamlit App
The final model is deployed in a Streamlit app for interactive predictions.

#### Run locally
```python
# clone repo
git clone https://github.com/riddhibajaj/insurance-charges-prediction.git
cd insurance-charges-prediction

# install dependencies
pip install -r requirements.txt

# run the app
streamlit run streamlit_app.py
```

#### Input Fields
* Age, Sex, BMI, Children, Smoker, Region

#### Output
* Predicted **insurance charges** (in USD).

## Project Structure
```python
├── streamlit_app.py         # Streamlit app
├── insurance.csv            # Dataset
├── model_pipeline.pkl       # Trained model
├── insurance_charges.ipynb  # EDA & experiments
├── requirements.txt         # Dependencies
└── README.md                # Project summary
```

## Results
* Final chosen model: Gradient Boosting Regressor (pipeline with preprocessing)
* Test set R² ≈ 0.87
* Captures strong effects of smoking, age, and BMI reliably.
* **Key takeaways**:
  * Medical costs are **not random**, lifestyle choices (like smoking) and health indicators (like BMI) strongly influence them.
  * This project shows how data science can **quantify these impacts** and build tools for decision-making. 

## Acknowledgments
* Dataset: Kaggle — Medical Cost Personal Dataset
* Libraries: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `streamlit`, `joblib`
<<<<<<< HEAD

## License
MIT License © 2025 Riddhi Bajaj
=======