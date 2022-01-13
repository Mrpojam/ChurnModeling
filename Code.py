import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("/content/Churn.csv")

data.drop("RowNumber", axis=1, inplace=True)
data.drop("CustomerId", axis=1, inplace=True)
data.drop("Surname", axis=1, inplace=True)


data['Gender'] = data['Gender'].replace({'Female':1,'Male':0})

data['Geography'].unique()

dummies = pd.get_dummies(data['Geography'])

data = pd.concat([data, dummies], axis=1)
data.drop('Geography', axis=1, inplace=True)
data.head()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = data.drop('Exited', axis=True)
Y = data['Exited']
X = sc.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


# ## Train your model

# In[42]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

models = [LogisticRegression(), RandomForestClassifier(), SVC(), GaussianNB()]

for model in models:
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred) * 100
  print (model, "Accuracy : ", accuracy)


error_rate = []
# Might take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


knn = KNeighborsClassifier(n_neighbors=16)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


knn.score(X_train, y_train)


knn.score(X_test, y_test)


import joblib

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
joblib.dump(rf, "my_random_forest.joblib")




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


ann = Sequential()
ann.add(Dense(12, activation="sigmoid"))
ann.add(Dense(48, activation="sigmoid"))
ann.add(Dense(24, activation="sigmoid"))
ann.add(Dense(12, activation="sigmoid"))
ann.add(Dense(1, activation="sigmoid"))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


from keras.utils.vis_utils import plot_model

ann.fit(X_train, y_train, epochs=250, batch_size=8)

ann.save("ann_model")

