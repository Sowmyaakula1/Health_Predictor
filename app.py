from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import plotly.express as px
import joblib

app = Flask(__name__)

# Load and prepare the data
df = pd.read_csv("health_fitness_dataset.csv")
df = df.dropna()

# Feature engineering
df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
df['activity_intensity'] = pd.Categorical(df['intensity']).codes

# Encode categorical variables
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['health_condition'] = le.fit_transform(df['health_condition'])
df['smoking_status'] = le.fit_transform(df['smoking_status'])

# Prepare features and target variable
features = [
    'age', 'gender', 'bmi', 'avg_heart_rate', 'resting_heart_rate',
    'blood_pressure_systolic', 'blood_pressure_diastolic', 'stress_level',
    'daily_steps', 'hours_sleep', 'activity_intensity', 'hydration_level',
    'smoking_status'
]
X = df[features]
y = df['health_condition']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Check if trained models already exist; if not, train them.
try:
    models = joblib.load('trained_models.pkl')
    print("Loaded trained models from file.")
except FileNotFoundError:
    print("Training models...")
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    # Fit models and store their accuracies
    for model_name, model in models.items():
        model.fit(X_scaled, y)
        predictions = model.predict(X_scaled)
        accuracy = model.score(X_scaled, y)
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        print(f"{model_name} Classification Report:\n", classification_report(y, predictions))
    
    # Save trained models to disk
    joblib.dump(models, 'trained_models.pkl')

@app.route('/')
def index():
    total_participants = df['participant_id'].nunique()
    avg_age = df['age'].mean()
    avg_bmi = df['bmi'].mean()
    most_common_activity = df['activity_type'].mode().values[0]

    activity_dist = df['activity_type'].value_counts()
    activity_fig = px.pie(values=activity_dist.values, names=activity_dist.index, title='Activity Distribution')
    activity_chart = activity_fig.to_html(full_html=False)

    return render_template('index.html', 
                           total_participants=total_participants,
                           avg_age=avg_age,
                           avg_bmi=avg_bmi,
                           most_common_activity=most_common_activity,
                           activity_chart=activity_chart)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Convert form data to appropriate numeric types
    input_data = np.array([[
        float(data['age']), int(data['gender']), float(data['bmi']), float(data['avg_heart_rate']),
        float(data['resting_heart_rate']), float(data['blood_pressure_systolic']),
        float(data['blood_pressure_diastolic']), int(data['stress_level']),
        int(data['daily_steps']), float(data['hours_sleep']), int(data['activity_intensity']),
        float(data['hydration_level']), int(data['smoking_status'])
    ]])
    
    input_scaled = scaler.transform(input_data)

    # Use the best performing model
    model_to_use = models["Random Forest"]
    prediction_proba = model_to_use.predict_proba(input_scaled)[0]
    
    risk_level = "High" if prediction_proba[1] > 0.5 else "Low"
    
    # Detailed feedback based on input data
    feedback = []
    if float(data['bmi']) > 30:
        feedback.append("Your BMI indicates obesity. Consider improving your diet and increasing physical activity.")
    if int(data['stress_level']) > 7:
        feedback.append("Your stress level is high. Practice mindfulness and relaxation techniques.")
    if float(data['hours_sleep']) < 6:
        feedback.append("You are not getting enough sleep. Aim for 7-9 hours of quality sleep per night.")
    if int(data['activity_intensity']) < 2:
        feedback.append("Your physical activity level is low. Try to incorporate more exercise into your routine.")
    if int(data['smoking_status']) == 3:
        feedback.append("Smoking is a significant health risk. Seek support to quit smoking.")
    
    if risk_level == "Low":
        feedback.append("Your health risk is low. Keep up the good work!")

    # Specific diseases at risk
    diseases_at_risk = []
    if float(data['bmi']) > 30:
        diseases_at_risk.append("Type 2 Diabetes, Cardiovascular Diseases")
    if int(data['smoking_status']) == 3:
        diseases_at_risk.append("Lung Cancer, Chronic Obstructive Pulmonary Disease (COPD)")
    if int(data['stress_level']) > 7:
        diseases_at_risk.append("Hypertension, Mental Health Disorders")

    return jsonify({
        'risk_probability': float(prediction_proba[1]),
        'risk_level': risk_level,
        'feedback': feedback,
        'diseases_at_risk': list(set(diseases_at_risk))  # Remove duplicates
    })

@app.route('/dashboard')
def dashboard():
    bmi_fig = px.scatter(df, x='bmi', y='avg_heart_rate', color='health_condition',
                         title='BMI vs Average Heart Rate', hover_data=['age'])

    sleep_fig = px.box(df, x='health_condition', y='hours_sleep',
                       title='Sleep Patterns by Health Condition')

    importance_fig = px.bar(x=features, y=models["Random Forest"].feature_importances_,
                             title='Feature Importance (Random Forest)', labels={'x':'Features', 'y':'Importance'})

    return render_template('dashboard.html',
                           bmi_chart=bmi_fig.to_html(full_html=False),
                           sleep_chart=sleep_fig.to_html(full_html=False),
                           importance_chart=importance_fig.to_html(full_html=False))

if __name__ == '__main__':
    app.run(debug=True)