import pandas as pd
import numpy as np
import csv

filename = '../Data/diabetes.csv'
from sklearn.model_selection import train_test_split

diabetes_data = pd.read_csv(filename)

diabetes_data.head(20)

print("Data Null \n", diabetes_data.isnull().sum())

print("Data Kosong \n", diabetes_data.empty)

print("Data Nan \n", diabetes_data.isna().sum())

diabetes_data.describe()

diabetes_data['Outcome'] = diabetes_data.Outcome.astype(int)

freq = diabetes_data.Outcome.value_counts()

freq.plot(kind='bar')

X = diabetes_data.drop('Outcome', axis=1)

y = diabetes_data.Outcome


#TRAINING

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state=0)


print("Bentuk X_train ", X_train.shape)
print("Bentuk X_test ", X_test.shape)

print("Bentuk y_train ", y_train.shape)
print("Bentuk y_test ", y_test.shape)


print("y_train ", y_train)
print("y_test ", y_test)

from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors=2, p=2)

from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier(random_state=0, max_depth=3)

KNN.fit(X_train, y_train)

DT.fit(X_train, y_train)


#KODE GPT BUAT FEATURE NAME
#import pandas as pd

# # Create a DataFrame with feature names
# feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
# X_new = pd.DataFrame([[6, 148, 72, 35, 0, 33.6, 0.627, 50]], columns=feature_names)

# # Predict using models
# knn_predict = KNN.predict(X_new)
# dt_predict = DT.predict(X_new)
# print("Label prediksi KNN:", knn_predict)
# print("Label prediksi Decision Tree:", dt_predict)


X_new = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

print("X_new yang akan diprediksi", X_new.shape)

knn_predict = KNN.predict(X_new)

print("Label prediksi KNN", knn_predict)

dt_predict = DT.predict(X_new)

print("Label prediksi DT", dt_predict)

y_pred_knn = KNN.predict(X_test)

y_pred_dt = DT.predict(X_test)

print("Hasil prediksi KNN pada X_test : ", y_pred_knn)

print("Hasil prediksi DT pada X_test : ", y_pred_dt)


print("Akurasi Model KNN dengan fungsi score ", round(KNN.score(X_test, y_test), 3))

print("Akurasi Model DT perbandingan prediksi vs label ", round(DT.score(X_test, y_test), 3))

import pickle 

with open('knn_dt_diabetes_model.pkl', 'wb') as f:
    pickle.dump((KNN, DT), f)
    
print("Model KNN dan DT berhasil Disimpan")