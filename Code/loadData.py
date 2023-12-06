import pandas as pd
import numpy as np
import os
import sklearn.model_selection as ms
import category_encoders as ce



def organizeDataIntoOne(mathData, portugueseData):
    # setting up column names 
    defaultColumnNames = mathData.loc[0, :].values.tolist()
    mathColumnNames = defaultColumnNames.copy()
    
    # setting up the combine
    mathColumnNames[-1] = 'MG3'
    mathColumnNames[-2] = 'MG2'
    mathColumnNames[-3] = 'MG1'

    portuColumnNames = defaultColumnNames.copy()
    portuColumnNames[-1] = 'PG3'
    portuColumnNames[-2] = 'PG2'
    portuColumnNames[-3] = 'PG1'

    mathData = mathData.iloc[1:]
    portugueseData = portugueseData.iloc[1:]

    # set up column names
    mathData.columns = mathColumnNames
    portugueseData.columns = portuColumnNames
    
    commonColumns = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"]

    # find all common student from both data set 
    data = pd.merge(portugueseData, mathData, on=commonColumns)
    data = data.where(pd.notna(data), None)

    # clean up samilar columns
    for eachColumn in data.columns:
        if '_y' in eachColumn:
            data = data.drop(eachColumn, axis=1)
        elif '_x' in eachColumn:
            data = data.rename(columns={eachColumn: eachColumn[:-2]})

    return data

def oneHot(data):

    # further modify data into numerical and categorical
    sampleData = data.iloc[:1].values.flatten()

    categoricalData = pd.DataFrame()

    columnsName = data.columns
    for i in range(len(columnsName)):
        try:
            num = int(sampleData[i])
        except ValueError:
            categoricalData[columnsName[i]] = data[columnsName[i]]
    
    # split the data set into training and teseting data
    x = data.drop(['MG3', 'PG3', 'MG1', 'MG2', 'PG1', 'PG2'], axis=1)
    target = data[['MG3','PG3']].copy().astype('int64')
    target = target['MG3'] + target['PG3']
    target = target.to_frame()
    xTrain, xTest, tTrain, tTest = ms.train_test_split(x, target, test_size=0.25, random_state=0)

    tTrain = (tTrain >= 20).astype(int)
    tTest = (tTest >= 20).astype(int)

    # one hot encoding
    encoder = ce.OneHotEncoder(cols=categoricalData)
    xTrain = encoder.fit_transform(xTrain)
    xTest = encoder.transform(xTest)
    
    # create features
    features = encoder.feature_names_out_
    f = open('features.txt', 'w')
    for i in features:
        f.write(i + '\n')
    f.close()

    # create mappings for categorical columns
    mapping = encoder.category_mapping
    f = open('mapping.txt', 'w')
    for i in range(len(mapping)):
        f.write(str(mapping[i]))
    f.close()

    # save to txt for linear regression
    np.savetxt('xTrain.txt', xTrain, fmt='%s')
    np.savetxt('xTest.txt', xTest, fmt='%s')
    np.savetxt('tTrain.txt', tTrain, fmt='%s')
    np.savetxt('tTest.txt', tTest, fmt='%s')

    return xTrain, xTest, tTrain, tTest, data.columns.tolist()
