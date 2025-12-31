import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

parkinson = pd.read_csv("/content/parkinsons.csv")
parkinson = parkinson.dropna()

X = parkinson[['DFA', 'PPE']]
y = parkinson['status']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

svc = SVC()
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

import joblib

joblib.dump(svc, 'my_model.joblib')
