
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# Load dataset and prepare it the same way
@st.cache_data
def load_data():
    df = pd.read_csv('titanic.csv')
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop(['Cabin'], axis=1, inplace=True)
    
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['IsAlone'] = np.where(df['FamilySize'] == 0, 1, 0)
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 80], labels=[0, 1, 2, 3, 4])
    df['FareBand'] = pd.qcut(df['Fare'], 4, labels=[0, 1, 2, 3])
    
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    df.drop(['PassengerId', 'Name', 'Ticket', 'Age', 'Fare'], axis=1, inplace=True)
    
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GaussianNB()
    model.fit(X_scaled, y)
    
    return model, scaler

model, scaler = load_data()

# App UI
st.title("🚢 Titanic Survival Prediction App (Naive Bayes)")

st.write("Enter passenger details to check if they'd survive...")

# Input form
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ['Male', 'Female'])
sibsp = st.slider("Siblings/Spouse Aboard", 0, 8, 0)
parch = st.slider("Parents/Children Aboard", 0, 6, 0)
embarked = st.selectbox("Embarked Port", ['S', 'C', 'Q'])
age_group = st.selectbox("Age Group", ['Child (0-12)', 'Teen (13-17)', 'Young Adult (18-35)', 'Adult (36-60)', 'Elderly (60+)'])
fare_band = st.selectbox("Fare Band", ['Low (Q1)', 'Medium (Q2)', 'High (Q3)', 'Very High (Q4)'])

# Encode inputs
sex_code = 0 if sex == 'Male' else 1
embarked_code = {'S': 0, 'C': 1, 'Q': 2}[embarked]
age_code = ['Child (0-12)', 'Teen (13-17)', 'Young Adult (18-35)', 'Adult (36-60)', 'Elderly (60+)'].index(age_group)
fare_code = ['Low (Q1)', 'Medium (Q2)', 'High (Q3)', 'Very High (Q4)'].index(fare_band)

family_size = sibsp + parch
is_alone = 1 if family_size == 0 else 0

# Final input for prediction
user_input = pd.DataFrame([{
    'Pclass': pclass,
    'Sex': sex_code,
    'SibSp': sibsp,
    'Parch': parch,
    'Embarked': embarked_code,
    'FamilySize': family_size,
    'IsAlone': is_alone,
    'AgeGroup': age_code,
    'FareBand': fare_code
}])

# Prediction
user_scaled = scaler.transform(user_input)
prediction = model.predict(user_scaled)[0]

# Output
st.subheader("🔮 Prediction Result:")
if prediction == 1:
    st.success("✅ Survived")
else:
    st.error("❌ Did NOT Survive")
