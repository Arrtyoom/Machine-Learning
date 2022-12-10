import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], 1))  # attribute
y = np.array(data[predict])  # labels

# x_train and y train will be x and y split up
# x_test and y_test will be what test de accuracy of our model (x and y)
# test_size="0.1" will take 10% of our model value
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

linear = linear_model.LinearRegression()  # make calculation to make a linear regression

# will search the best fit line using x_train and y_train and put it in linear
linear.fit(x_train, y_train)  # actual
acc = linear.score(x_test, y_test)  # expected
print(acc)


print('Coefficient: \n', linear.coef_)

# we receive 5 coefficient because we have
# 5 information so the line is in 5 dimension
# if we would have to info (ex: time, grade) we
# would have had a 2 dimensional line


print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
