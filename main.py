import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

parkinson = pd.read_csv("/content/parkinsons.csv")
parkinson = parkinson.dropna()

x = parkinson[['DFA', 'PPE']]
y = parkinson['status']

x_train_i, x_test_i, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train_i)
x_test = scaler.fit_transform(x_test_i)



svc = SVC()
svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

import joblib

joblib.dump(svc, 'my_model.joblib')
