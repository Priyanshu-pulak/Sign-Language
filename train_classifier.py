import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

with open('data.pkl', 'rb') as file:
    data_dict = pickle.load(file)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=120)
model = SVC()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)
print(f"{round(score * 100, 2)}% of samples were classified correctly !")

with open('model.pkl', 'wb') as file:
    pickle.dump({"model": model}, file)
