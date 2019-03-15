#Random Forest with Sklearn Datasets - Digits

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#================================================
#Load Data
#================================================

from sklearn.datasets import load_digits
digits = load_digits()

print(dir(digits))
print(digits['data'][0])
print(digits['images'][0])
print(digits['target'][0])

#================================================
#Plot Data
#================================================

# plt.figure('Digits', figsize = (5, 5))
# for i in range(10):
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(digits['images'][i], cmap = 'gray')

# plt.show()

#================================================
#Create DataFrame
#================================================

df = pd.DataFrame(digits['data'])
df['target'] = digits['target']
print(df.head(1))

#================================================
#Random Forest Algorithm
#================================================

#Split Train (90%) and Test (10%)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    df.drop(['target'], axis = 'columns'),      #use all columns in df without target column
    df['target'],
    test_size = .1
)
print(len(x_train))
print(len(x_test))

#Import Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators= 40)    #n_estimator = The number of trees in the forest (default value = 100, in version 0.22)

#Training Model
model.fit(x_train, y_train)

#Testing Model Accuracy
print(model.score(x_train, y_train) * 100, '%')

#Prediction
print(x_test.iloc[0])
print(model.predict([x_test.iloc[0]])[0])
print(y_test.iloc[0])

#================================================
#Plot Random Forest Prediction
#================================================

x_plot = x_test.iloc[0].values.reshape(8,8)
print(x_plot)

plt.figure('Digits', figsize = (5, 5))
plt.imshow(x_plot, cmap = 'gray')
plt.title('Actual: {} / Prediction: {} / Accuracy: {}'.format(
    y_test.iloc[0],
    model.predict([x_test.iloc[0]])[0],
    str(round(model.score(x_test, y_test)) * 100) + '%'
))

plt.show()
