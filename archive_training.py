import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

model = RandomForestClassifier()
data = pd.read_csv('Iris.csv')

X_data = data.drop('Species', axis=1)
y_data = data['Species']
X_ach_traning, X_testing, y_ach_traning, y_testing = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
model.fit(X_ach_traning, y_ach_traning)
dump(model, 'archive_model_traning.joblib')