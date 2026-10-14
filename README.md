# Email Campaign Effectiveness Prediction

## 🚀 Live Demo

**Try the deployed application:**  
[Email Campaign Effectiveness Prediction — Live App](https://email-campaign-effectiveness-prediction-k2qepzgqzv9gbf5i3d9vtv.streamlit.app/)

**GitHub Repository:**  
[View Source Code on GitHub](https://github.com/khadarbasha2212-gif/email-campaign-effectiveness-prediction)

---

## 📌 Project Overview

Email Campaign Effectiveness Prediction is a machine learning project designed to predict email campaign status based on email characteristics and campaign-related features.

The project follows a machine learning workflow that includes data understanding, data quality checks, preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and application deployment.

An interactive Streamlit web application allows users to enter email campaign details and obtain a predicted status class along with the model's class probabilities.

## 🎯 Project Objectives

- Understand and analyze email campaign data.
- Perform data cleaning and preprocessing.
- Conduct exploratory data analysis to identify patterns.
- Engineer features for machine learning.
- Train and compare multiple classification algorithms.
- Evaluate models on an imbalanced multiclass dataset.
- Build a reusable preprocessing and prediction pipeline.
- Deploy an interactive machine learning application.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Web Application | Streamlit |
| Model Persistence | Joblib |
| Development Environment | VS Code |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

## 📊 Dataset Description

The dataset contains email-related and campaign-related features, including:

- Email type
- Subject hotness score
- Email source type
- Customer location
- Email campaign type
- Total past communications
- Email sending time category
- Word count
- Total links
- Total images

The target variable is `Email_Status`, which contains three numeric classes: `0`, `1`, and `2`.

The exact business meanings of these status classes have not been verified from the original dataset documentation, so they are retained as numeric labels.

## 🔍 Project Workflow

### 1. Data Understanding
- Examined dataset dimensions and column names.
- Analyzed target class distribution.
- Reviewed feature types and basic statistics.

### 2. Data Quality and Cleaning
- Checked for missing values and duplicate records.
- Handled missing categorical and numerical values.
- Prepared a cleaned dataset for analysis.

### 3. Exploratory Data Analysis (EDA)
- Visualized email status distribution.
- Analyzed campaign status patterns.
- Explored customer location and subject hotness.
- Examined relationships between word count and email status.

### 4. Feature Engineering
- Removed the identifier column from the model features.
- Encoded categorical variables.
- Prepared features for multiclass classification.

### 5. Model Training and Comparison

The following models were evaluated:

- Logistic Regression
- Scaled Logistic Regression
- Random Forest
- XGBoost
- Class-weighted XGBoost

### 6. Model Evaluation
Models were compared using accuracy, balanced accuracy, macro F1-score, confusion matrices, and class-specific precision and recall.

### 7. Model Pipeline
A reusable pipeline was created to apply preprocessing and generate predictions consistently.

### 8. Application Development and Deployment
- Developed an interactive Streamlit application.
- Integrated the saved machine learning pipeline.
- Displayed predicted status classes and class probabilities.
- Deployed the application using Streamlit Community Cloud.

## 🤖 Selected Model

**Class-weighted XGBoost Classifier**

Class weighting was used to help address the dataset's class imbalance. The model was selected for the current application based on the evaluation experiments.

## 📈 Model Evaluation Results

The selected model achieved the following results on the held-out test dataset:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 64.73% |
| Balanced Accuracy | 53.32% |
| Macro F1-score | 0.4282 |

### Interpretation

- **Accuracy:** 64.73% of test examples were classified correctly.
- **Balanced accuracy:** 53.32%, reflecting average recall across the three classes.
- **Macro F1-score:** 0.4282, measuring the average F1-score across classes with equal weight.

Because the dataset is imbalanced, accuracy alone does not fully describe model performance. Minority-class precision and recall require further improvement.

## ✨ Application Features

- Interactive email campaign input form.
- Three-class status prediction.
- Class probability display.
- Reusable machine learning pipeline.
- Browser-based user interface.
- Publicly accessible live demonstration.

## 📁 Project Structure

```text
email_campaign_effectiveness_prediction/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── weighted_xgboost_pipeline.joblib
├── notebooks/
│   ├── 01_data_understanding.py
│   ├── 02_data_quality.py
│   ├── 03_data_cleaning.py
│   ├── 04_eda.py
│   ├── 05_feature_engineering.py
│   ├── 06_train_test_split.py
│   └── ...
└── reports/
    ├── model_comparison.csv
    └── evaluation plots
```

*Note: The dataset and generated artifacts are not all tracked in GitHub. The trained pipeline is included for deployment.*

## 💻 Run the Project Locally

### Prerequisites

- Python installed
- Git installed
- Visual Studio Code or another Python development environment

### Step 1: Clone the Repository

```bash
git clone https://github.com/khadarbasha2212-gif/email-campaign-effectiveness-prediction.git
cd email-campaign-effectiveness-prediction
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment on Windows

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will display a local URL, usually:

`http://localhost:8501`

**Note:** The trained model must be available at `models/weighted_xgboost_pipeline.joblib` before the application can generate predictions. The raw dataset is needed if you want to rerun the complete training workflow.

## 🔮 Future Improvements

- Improve minority-class precision and recall.
- Perform additional hyperparameter tuning.
- Compare class-weighting strategies and resampling methods.
- Evaluate probability calibration.
- Add automated unit and integration tests.
- Add API-based prediction support.
- Implement prediction monitoring and model drift detection.
- Document the business meaning of each target class.
- Evaluate the model on additional datasets.

## ⚠️ Limitations

This project is intended for learning and portfolio demonstration.

The reported metrics are specific to the current dataset and test split. They do not guarantee performance on new data or real-world email campaigns.

The model's predictions should not be interpreted as email opens, clicks, conversions, or other business outcomes until the target class definitions have been verified.

## 👨‍💻 Author

**Khadar Basha**

GitHub Profile: [khadarbasha2212-gif](https://github.com/khadarbasha2212-gif)

## 🔗 Project Links

- **Live Application:** [Launch the Streamlit App](https://email-campaign-effectiveness-prediction-k2qepzgqzv9gbf5i3d9vtv.streamlit.app/)
- **Source Code:** [GitHub Repository](https://github.com/khadarbasha2212-gif/email-campaign-effectiveness-prediction)

---

If you find this project useful, feel free to explore the repository and try the live application.