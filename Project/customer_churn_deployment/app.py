```python
from flask import Flask, render_template, request
import os
import uuid
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

app = Flask(__name__)

MODEL_FILE = 'customer_churn_elastic_net.pkl'
DATA_FILE = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

# Numerical features
numerical_features = [
    'SeniorCitizen',
    'tenure',
    'MonthlyCharges',
    'TotalCharges'
]

# Categorical features
categorical_features = [
    'gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod'
]

features = numerical_features + categorical_features

# Dataset download sources used only if the local CSV is missing
DATA_URLS = [
    'https://raw.githubusercontent.com/Giskard-AI/examples/main/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv',
    'https://raw.githubusercontent.com/aiplanethub/Datasets/master/WA_Fn-UseC_-Telco-Customer-Churn.csv'
]


# --------------------------------------------------
# TRAIN MODEL IF THE SAVED MODEL DOES NOT EXIST
# --------------------------------------------------
def train_model():

    if not os.path.exists(DATA_FILE):
        last_error = None

        for url in DATA_URLS:
            try:
                df = pd.read_csv(url)
                df.to_csv(DATA_FILE, index=False)
                print("Dataset downloaded successfully.")
                break

            except Exception as e:
                last_error = e

        else:
            raise RuntimeError(
                f"Could not download dataset: {last_error}"
            )

    else:
        df = pd.read_csv(DATA_FILE)

    # Clean TotalCharges
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)

    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'],
        errors='coerce'
    )

    df['TotalCharges'] = df['TotalCharges'].fillna(
        df['TotalCharges'].median()
    )

    # Select input features and target
    model_df = df[features + ['Churn']].copy()

    X = model_df[features]
    y = model_df['Churn'].map({
        'No': 0,
        'Yes': 1
    })

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                'num',
                StandardScaler(),
                numerical_features
            ),
            (
                'cat',
                OneHotEncoder(handle_unknown='ignore'),
                categorical_features
            )
        ]
    )

    # Elastic Net Logistic Regression
    classifier = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0.5,
        C=1.0,
        max_iter=5000,
        random_state=42
    )

    # Complete machine learning pipeline
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', classifier)
        ]
    )

    # Train the model
    pipeline.fit(X, y)

    # Save the trained model
    joblib.dump(pipeline, MODEL_FILE)

    print("Elastic Net model trained and saved successfully.")

    return pipeline


# --------------------------------------------------
# LOAD SAVED MODEL OR TRAIN A NEW MODEL
# --------------------------------------------------
def get_model():

    if os.path.exists(MODEL_FILE):
        print("Loading saved Elastic Net model.")
        return joblib.load(MODEL_FILE)

    print("Saved model not found. Training a new model.")
    return train_model()


model = get_model()


# --------------------------------------------------
# CLASSIFY CHURN RISK
# --------------------------------------------------
def classify_risk(probability):

    if probability < 30:
        return 'Low Risk'

    elif probability < 60:
        return 'Medium Risk'

    else:
        return 'High Risk'


# --------------------------------------------------
# HOME PAGE AND CUSTOMER CHURN PREDICTION
# --------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def home():

    result = None
    probability = None
    risk_level = None
    error = None

    form_data = (
        request.form.to_dict()
        if request.method == 'POST'
        else {}
    )

    # Keep a prediction ID stable when the same form
    # submission is refreshed.
    prediction_id = (
        request.form.get('prediction_id') or str(uuid.uuid4())
    )

    if request.method == 'POST':

        try:
            # Collect customer input
            data = {
                'SeniorCitizen': int(
                    request.form['SeniorCitizen']
                ),

                'tenure': float(
                    request.form['tenure']
                ),

                'MonthlyCharges': float(
                    request.form['MonthlyCharges']
                ),

                'TotalCharges': float(
                    request.form['TotalCharges']
                ),

                'gender': request.form['gender'],
                'Partner': request.form['Partner'],
                'Dependents': request.form['Dependents'],
                'PhoneService': request.form['PhoneService'],
                'MultipleLines': request.form['MultipleLines'],
                'InternetService': request.form['InternetService'],
                'OnlineSecurity': request.form['OnlineSecurity'],
                'OnlineBackup': request.form['OnlineBackup'],
                'DeviceProtection': request.form['DeviceProtection'],
                'TechSupport': request.form['TechSupport'],
                'StreamingTV': request.form['StreamingTV'],
                'StreamingMovies': request.form['StreamingMovies'],
                'Contract': request.form['Contract'],
                'PaperlessBilling': request.form['PaperlessBilling'],
                'PaymentMethod': request.form['PaymentMethod']
            }

            # Convert input into a DataFrame
            input_df = pd.DataFrame(
                [data],
                columns=features
            )

            # Predict churn
            prediction = int(
                model.predict(input_df)[0]
            )

            # Get estimated churn probability as a percentage
            probability = float(
                model.predict_proba(input_df)[0][1]
            ) * 100

            # Ensure probability stays within 0-100
            probability = max(
                0.0,
                min(100.0, probability)
            )

            # Binary prediction result
            if prediction == 1:
                result = 'Customer Likely to Churn'
            else:
                result = 'Customer Not Likely to Churn'

            # Three-level risk classification
            risk_level = classify_risk(probability)

        except (KeyError, ValueError, TypeError) as e:
            error = (
                'Please enter valid values for all required '
                f'customer fields. Details: {e}'
            )

        except Exception as e:
            app.logger.exception(
                'Customer churn prediction failed.'
            )
            error = (
                'Unable to generate the prediction. '
                'Please check the inputs and try again.'
            )

    return render_template(
        'index.html',
        result=result,
        probability=probability,
        risk_level=risk_level,
        error=error,
        form_data=form_data,
        prediction_id=prediction_id
    )


# --------------------------------------------------
# HEALTH CHECK FOR RENDER
# --------------------------------------------------
@app.route('/health')
def health():

    return {
        'status': 'ok',
        'model': 'Elastic Net Logistic Regression'
    }


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------
if __name__ == '__main__':

    port = int(os.environ.get('PORT', 5000))

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )
```
