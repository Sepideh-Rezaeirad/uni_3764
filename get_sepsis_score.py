#!/usr/bin/env python
from sklearn.externals import joblib
import numpy as np
import pandas as pd

def get_sepsis_score(data, model):
    num_rows = len(data)
    meanF = np.load('meanF.npy')
    ####### Impute
    # imputePatient = []
    interval = range(data.shape[0])

    # for interval in range(data.shape[0]):      #### loop for on intervals
    if interval == 0:
        newData = np.copy(data[0,:])
        for column in range(40):       ########  loop for on columns
            if (np.isnan(newData[column])):
                newData[column] = meanF[column]


    else:
        newData = np.copy(data[0,:])
        for column in range(40):       ########  loop for on columns
            if (np.isnan(newData[column])):
                newData[column] = meanF[column]
        data[0, :] = newData
        df = pd.DataFrame.from_records(data)
        df.interpolate(method='linear', inplace=True)
        newData1 = np.array(df)


    data = newData1
    ####### End Impute
    if num_rows==1:
        label = 0.0
        score = 0.4
    else:
        M2 = joblib.load('model-saved_5_6.pkl')
        predicted2 = M2.predict(data[[-2, -1], :])

        M3 = joblib.load('model-saved_6_6.pkl')
        predicted3 = M3.predict(data[[-2, -1], :])

        new_data = np.zeros((np.size(predicted3, 0), 2))
        new_data[:, 0] = predicted2
        new_data[:, 1] = predicted3

        M = joblib.load('model-saved.pkl')
        predicted = M.predict(new_data)

        #######################################################################

        # predicted = M1.predict(data[[-2, -1], :])
        if predicted[-1] == 0:
            score = 0.4
        else:
            score = 0.6
        label = predicted[-1]


    return score, label

def load_sepsis_model():

    return None
