# 🫀 Heart Disease Classification — ML Assignment 2

**Course:** Machine Learning | M.Tech (AIML/DSE) | BITS Pilani WILP

---

## a. Problem Statement

Predict the presence or absence of heart disease in a patient based on 13 clinical and diagnostic features using multiple supervised classification models. This is a **binary classification** problem where the target variable indicates:
- `1` → Heart Disease Present
- `0` → No Heart Disease

---

## b. Dataset Description

| Property | Value |
|----------|-------|
| **Name** | Heart Disease Dataset |
| **Source** | [Kaggle — Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) |
| **Original Source** | UCI Machine Learning Repository |
| **Instances** | 302 (after removing duplicates) |
| **Features** | 13 predictive features + 1 target |
| **Task** | Binary Classification |

### Features

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `age` | Numeric | Age in years |
| 2 | `sex` | Categorical | 1=Male, 0=Female |
| 3 | `cp` | Categorical | Chest pain type (0-3) |
| 4 | `trestbps` | Numeric | Resting blood pressure (mm Hg) |
| 5 | `chol` | Numeric | Serum cholesterol (mg/dl) |
| 6 | `fbs` | Categorical | Fasting blood sugar > 120 mg/dl |
| 7 | `restecg` | Categorical | Resting ECG results |
| 8 | `thalach` | Numeric | Max heart rate achieved |
| 9 | `exang` | Categorical | Exercise-induced angina |
| 10 | `oldpeak` | Numeric | ST depression (exercise vs rest) |
| 11 | `slope` | Categorical | Slope of peak exercise ST segment |
| 12 | `ca` | Numeric | Major vessels colored by fluoroscopy (0-3) |
| 13 | `thal` | Categorical | Thalassemia type |

---

## c. GitHub Repository Link

> **TODO:** Replace with your actual GitHub repository link after pushing the code.
>
> `https://github.com/<your-username>/<repo-name>`

---

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.8033 | 0.8712 | 0.8000 | 0.8485 | 0.8235 | 0.6031 |
| Decision Tree | 0.8033 | 0.8019 | 0.8182 | 0.8182 | 0.8182 | 0.6039 |
| KNN | 0.7869 | 0.8377 | 0.7778 | 0.8485 | 0.8116 | 0.5702 |
| Naive Bayes | 0.7869 | 0.8842 | 0.8333 | 0.7576 | 0.7937 | 0.5771 |
| Random Forest | 0.7541 | 0.8588 | 0.7647 | 0.7879 | 0.7761 | 0.5038 |

---

### Model Observations

| ML Model Name | Observation about model performance |
|---------------|------------------------------------|
| Logistic Regression | Achieved good accuracy of 80.33%. Shows well-balanced precision and recall. Good AUC (0.8712), reasonable discriminative ability. Moderate MCC (0.6031) indicates acceptable performance. |
| Decision Tree | Achieved good accuracy of 80.33%. Shows well-balanced precision and recall. Good AUC (0.8019), reasonable discriminative ability. Moderate MCC (0.6039) indicates acceptable performance. |
| KNN | Achieved moderate accuracy of 78.69%. Higher recall (84.85%) than precision (77.78%), better at capturing positive cases. Good AUC (0.8377), reasonable discriminative ability. Moderate MCC (0.5702) indicates acceptable performance. |
| Naive Bayes | Achieved moderate accuracy of 78.69%. Higher precision (83.33%) than recall (75.76%), suggesting fewer false positives. Good AUC (0.8842), reasonable discriminative ability. Moderate MCC (0.5771) indicates acceptable performance. |
| Random Forest | Achieved moderate accuracy of 75.41%. Shows well-balanced precision and recall. Good AUC (0.8588), reasonable discriminative ability. Moderate MCC (0.5038) indicates acceptable performance. |
| **Overall Winner** | **Logistic Regression** — Best F1 Score (0.8235), providing the best balance of precision and recall on the Heart Disease dataset. |

---

## Streamlit App

> `https://2025ac05620mlassignment2.streamlit.app`

### App Features
- ✅ CSV file upload for test data
- ✅ Model selection dropdown (5 models)
- ✅ Display of all 6 evaluation metrics
- ✅ Confusion matrix visualization
- ✅ Classification report
- ✅ ROC Curve
- ✅ All-models comparison table with visual bar chart

---

## Repository Structure

```
project-folder/
│── app.py                    # Streamlit web application
│── requirements.txt          # Python dependencies
│── README.md                 # This file
│── test_data.csv             # Test data for evaluation
│── model/
│   ├── scaler.pkl            # Fitted StandardScaler
│   ├── feature_names.pkl     # Feature column names
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   └── random_forest.pkl
│── ML_Assignment_2.ipynb     # Colab notebook with full analysis
```

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Deployment on Streamlit Community Cloud

1. Push all files to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Click "New App" → Select your repo → Choose `app.py` → Deploy
