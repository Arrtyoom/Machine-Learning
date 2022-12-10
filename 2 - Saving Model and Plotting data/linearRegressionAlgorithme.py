import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], 1))  # attribute
y = np.array(data[predict])  # labels

# x_train and y train will be x and y split up
# x_test and y_test will be what test de accuracy of our model (x and y)
# test_size="0.1" will take 10% of our model value
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

""" # this part of the programme find the model with the greatest accuracy and save it 
    # it will be repeated once so we don't train our algorithme each time
best = 0
for _ in range(30):
    
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()  # make calculation to make a linear regression

    # will search the best fit line using x_train and y_train and put it in linear
    linear.fit(x_train, y_train)  # actual
    acc = linear.score(x_test, y_test)  # expected
    print(acc)

    if acc > best:
        best = acc
        # when a model that has a good accuracy is found we want to save it somewhere
        # instead of retraining our algorithme each time because it can be very long
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)  # save a model in a file in .pickle
"""


pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)  # load our model in the variable linear

print('Coefficient: \n', linear.coef_)

# we receive 5 coefficient because we have
# 5 information so the line is in 5 dimension
# if we would have to info (ex: time, grade) we
# would have had a 2 dimensional line


print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


# PLOT ON A GRID

p = "absences"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
