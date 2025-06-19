📘 Titanic Survival Prediction (Naive Bayes Classifier)
This is a mini machine learning project that predicts whether a passenger survived the Titanic disaster, based on attributes such as age, gender, ticket class, and more. The model uses the Naive Bayes algorithm for classification.


🚀 Features
-Data cleaning and preprocessing
-Exploratory Data Analysis (EDA) with visualizations
-Naive Bayes classification model
-Model evaluation and accuracy improvement
-Interactive Streamlit app for user input


📁 Files in the Repository
| File Name                           | Description                                                 |
| ----------------------------------- | ----------------------------------------------------------- |
| `Titanic_survival_prediction.ipynb` | Main Jupyter notebook for training and evaluating the model |
| `titanic_nb_app.py`                 | Streamlit app for live prediction using user inputs         |


📊 Technologies Used
-Python
-Pandas, NumPy
-Seaborn, Matplotlib
-Scikit-learn
-Streamlit


🧠 How It Works
The model is trained on the Titanic dataset (from Kaggle) with features such as:
-Pclass (Passenger class)
-Sex
-Age
-SibSp (Number of siblings/spouses aboard)
-Parch (Number of parents/children aboard)
-Fare
-Embarked
After preprocessing, a Naive Bayes classifier is trained and tested for accuracy.


💻 How to Run the Streamlit App
pip install streamlit
streamlit run titanic_nb_app.py


📈 Sample Visualizations
-Correlation heatmap
-Survival distribution by gender and class
-Age and fare histograms


✅ Project Goal
To demonstrate a simple ML pipeline using Naive Bayes and deploy an interactive prediction app for users with no coding knowledge.
