import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
#  In this file we are training our classifier which is Random Forest Classifier
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)


model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


score = accuracy_score(y_pred,y_test)
print('Accuracy score: ', score)

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)