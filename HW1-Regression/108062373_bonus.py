#!/usr/bin/env python
# coding: utf-8

# import packages
# Note: You cannot import any other packages!
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import random


# Global attributes
# Do not change anything here except TODO 1
StudentID = '108062373'  # TODO 1 : Fill your student ID here
# Please name your input csv file as 'input.csv'
input_dataroot = 'input_bonus.csv'
# Output file will be named as '[StudentID]_basic_prediction.csv'
output_dataroot = StudentID + '_bonus_prediction.csv'

input_datalist = []  # Initial datalist, saved as numpy array
output_datalist = []  # Your prediction, should be 20 * 2 matrix and saved as numpy array
# The format of each row should be [Date, TSMC_Price_Prediction]
# e.g. ['2021/10/15', 512]

# You can add your own global attributes here
x_datalist = []
y_datalist = []
validation_datalist = []
validation_y = []
testing_datalist = []
w = []

# Read input csv to datalist
with open(input_dataroot, newline='') as csvfile:
    input_datalist = np.array(list(csv.reader(csvfile)))

# From TODO 2 to TODO 6, you can declare your own input parameters, local attributes and return parameters


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def SplitData():
    # TODO 2: Split data, 2021/10/15 ~ 2021/11/11 for testing data, and the other for training data and validation data
    global training_datalist, validation_datalist, testing_datalist
    training_datalist = input_datalist[0:646]
    validation_datalist = input_datalist[647:674]
    testing_datalist = input_datalist[674:]


def PreprocessData():
    # TODO 3: Preprocess your data  e.g. split datalist to x_datalist and y_datalist
    # print(training_datalist)
    global x_datalist, y_datalist, validation_datalist, training_datalist
    for idx, L in enumerate(training_datalist):
        L[1] = L[1].replace(",", "")

        if idx >= 5:
            x_datalist.append([1.00,  float(training_datalist[idx-1][1]), float(training_datalist[idx-2][1]), float(
                training_datalist[idx-3][1]), float(training_datalist[idx-4][1]), float(training_datalist[idx-5][1])])
            y_datalist.append(float(L[1]))

    for L in validation_datalist:
        L[1] = float(L[1].replace(",", ""))
        validation_y.append(L[1])


def Regression():
    # TODO 4: Implement regression
    global x_datalist, w
    x_datalist = np.array(x_datalist)
    w = np.matmul(np.matmul(np.linalg.inv(
        np.matmul(x_datalist.T, x_datalist)), x_datalist.T), y_datalist)


def CountLoss():
    # TODO 5: Count loss of training and validation data
    global w, validation_datalist
    MAPE = 0
    num = 647
    for idx, data in enumerate(validation_y):
        x = []
        x.append(np.asarray(
            float(1), dtype='float64'))
        x.append(np.asarray(
            float(input_datalist[num-1-idx][1]), dtype='float64'))
        x.append(np.asarray(
            float(input_datalist[num-2-idx][1]), dtype='float64'))
        x.append(np.asarray(
            float(input_datalist[num-3-idx][1]), dtype='float64'))
        x.append(np.asarray(
            float(input_datalist[num-4-idx][1]), dtype='float64'))
        x.append(np.asarray(
            float(input_datalist[num-5-idx][1]), dtype='float64'))
        x = np.array(x)

        y = np.asarray(float(data), dtype='float64')
        y_predict = np.matmul(x.T, w)

        MAPE = MAPE + abs((y_predict - y)/y)

    MAPE = MAPE / len(validation_datalist) * 100
    print("MAPE = ", MAPE, "%")


def MakePrediction():
    # TODO 6: Make prediction of testing data
    global w, output_datalist, testing_datalist
    num = 674
    # print(input_datalist[674])
    for idx, data in enumerate(testing_datalist):
        x = []
        x.append(np.asarray(
            float(1), dtype='float64'))
        x.append(np.asarray(
            float(input_datalist[num-1-idx][1]), dtype='float64'))
        x.append(np.asarray(
            float(input_datalist[num-2-idx][1]), dtype='float64'))
        x.append(np.asarray(
            float(input_datalist[num-3-idx][1]), dtype='float64'))
        x.append(np.asarray(
            float(input_datalist[num-4-idx][1]), dtype='float64'))
        x.append(np.asarray(
            float(input_datalist[num-5-idx][1]), dtype='float64'))
        x = np.array(x)

        y_predict = np.matmul(x.T, w)

        output_datalist.append([data[0], y_predict])
        input_datalist[num-idx][1] = y_predict
    # print(output_datalist)


# TODO 7: Call functions of TODO 2 to TODO 6, train the model and make prediction
SplitData()
PreprocessData()
Regression()
CountLoss()
MakePrediction()

with open(output_dataroot, 'w', newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    #writer.writerow(['Date', 'TSMC Price'])

    for row in output_datalist:
        writer.writerow(row)
