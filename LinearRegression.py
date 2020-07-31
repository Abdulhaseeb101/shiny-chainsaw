
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn import model_selection
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

# Reading the dataset from the csv file:
dataSet = pd.read_csv('student-mat.csv', sep=';')

# Slicing the dataset to be more concise:
dataSet = dataSet[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

# The label that we are going to predict:
predict = 'G3'

# Exempting the prediction label out of the original dataset:
X = np.array(dataSet.drop([predict], 1))
Y = np.array(dataSet[predict])

xTrain, xTest, yTrain, yTest = model_selection.train_test_split(X, Y, test_size=0.1)

'''
bestModelScore = 0
for i in range(30):
        # Splitting the dataset into test dataset and training dataset:
        xTrain, xTest, yTrain, yTest = model_selection.train_test_split(X, Y, test_size=0.1)

        # THIS PART IS EXEMPTED TO PREVENT CREATING NEW MODELS EACH TIME

        # Creating the linear regression model and fit a line to the data:
        LinearRegression = linear_model.LinearRegression()
        LinearRegression.fit(xTrain, yTrain)
        accuracy = LinearRegression.score(xTest, yTest)
        print(accuracy)
        
        if accuracy > bestModelScore:
                # Save Prediction model:
                with open('studentModel.pickle', 'wb') as f:
                        pickle.dump(LinearRegression, f)
'''

pickleIn = open('studentModel.pickle', 'rb')
LinearRegression = pickle.load(pickleIn)

# Predict student grades with test data:
predictions = LinearRegression.predict(xTest)

for i in range(len(predictions)):
	print(predictions[i], xTest[i], yTest[i])

xAxis = 'G1'
style.use('ggplot')
plt.scatter(dataSet[xAxis], dataSet[predict])
plt.xlabel('First semester grades')
plt.ylabel('Final semester grades')
plt.show()
