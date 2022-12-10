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
x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


