import loadData
import pandas as pd
import numpy as np
import sklearn.model_selection
import sklearn.tree
import sklearn.metrics
import sklearn.naive_bayes
import sklearn.inspection
import sklearn.ensemble
import sklearn.linear_model
import sklearn.discriminant_analysis
import time
import LinearRegression
import generateReport

# decision tree
def decisionTree(xTrain, xTest, tTrain, tTest, maxIter):
    # trian data split for validation data
    spt = int(xTrain.shape[0] * 0.9)

    xVal = xTrain[spt:]
    tVal = tTrain[spt:]

    xTrain = xTrain[:spt]
    tTrain = tTrain[:spt]

    bestAcc = 0
    bestDtc = None
    
    # tune dtc
    for i in range(1, maxIter):
        dtc = sklearn.tree.DecisionTreeClassifier(max_depth=i)
        dtc.fit(xTrain,tTrain)

        tHat = dtc.predict(xVal)
        acc = sklearn.metrics.accuracy_score(tVal, tHat)

        if acc > bestAcc:
            bestAcc = acc
            bestDtc = dtc
    
    tTestHat = bestDtc.predict(xTest)
    trainedScore = sklearn.metrics.accuracy_score(tTest, tTestHat)

    return bestAcc, bestDtc, trainedScore, bestDtc.feature_importances_

def randomForest(xTrain, xTest, tTrain, tTest, maxIter):
    # trian data split for validation data
    spt = int(xTrain.shape[0] * 0.9)

    xVal = xTrain[spt:]
    tVal = tTrain[spt:]

    xTrain = xTrain[:spt]
    tTrain = tTrain[:spt]
    
    bestAcc = 0
    bestRfc = None
    
    # tune rfc
    for i in range(1, maxIter):
        rfc = sklearn.ensemble.RandomForestClassifier(max_depth=i)
        rfc.fit(xTrain, tTrain.values.ravel())

        tHat = rfc.predict(xVal)
        acc = sklearn.metrics.accuracy_score(tVal, tHat)

        if acc > bestAcc:
            bestAcc = acc
            bestRfc = rfc
    
    tTestHat = bestRfc.predict(xTest)
    trainedScore = sklearn.metrics.accuracy_score(tTest, tTestHat)

    return bestAcc, bestRfc, trainedScore, bestRfc.feature_importances_

def naiveBayes(xTrain, xTest, tTrain, tTest, maxIter):
    # trian data split for validation data
    spt = int(xTrain.shape[0] * 0.9)

    xVal = xTrain[spt:]
    tVal = tTrain[spt:]

    xTrain = xTrain[:spt]
    tTrain = tTrain[:spt]
    
    bestAcc = 0
    bestGnb = None

    # tune gnb
    for i in range(1, maxIter):
        gnb = sklearn.naive_bayes.GaussianNB()
        gnb.fit(xTrain, tTrain.values.ravel())

        tHat = gnb.predict(xVal)
        acc = sklearn.metrics.accuracy_score(tVal, tHat)

        if acc > bestAcc:
            bestAcc = acc
            bestGnb = gnb
    
    tTestHat = bestRfc.predict(xTest).ravel()
    trainedScore = sklearn.metrics.accuracy_score(tTest, tTestHat)
    c = sklearn.inspection.permutation_importance(bestGnb, xTest, tTestHat)

    return bestAcc, bestGnb, trainedScore, c.get('importances_mean')

def WQDA(xTrain, xTest, tTrain, tTest, maxIter):
    WQDA = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()
    WQDA.fit(xTrain, tTrain.values.ravel())

    tHat = WQDA.predict(xTest).ravel()
    acc = sklearn.metrics.accuracy_score(tTest, tHat)

    c = sklearn.inspection.permutation_importance(WQDA, xTest, tHat)
    return acc, c.get('importances_mean')

if (__name__ == '__main__'):
    # for total run time
    tst = time.time()

    # load in data and give it to loadData for further handling
    mathData = pd.read_csv('student-mat.csv', sep=";", header=None)
    portugueseData = pd.read_csv('student-por.csv', sep=';', header=None)

    data = loadData.organizeDataIntoOne(mathData, portugueseData)
    xTrain, xTest, tTrain, tTest, columnNames = loadData.oneHot(data)

    # default setting
    maxIter = 100
    scoreList = []

    print('Data Loaded...... Max iteration is set to be 100 ...... Start running Algrithoms :)\n')
    
    print('Decision Tree running ......')
    st = time.time()
    bestAcc, bestDtc, trainedScore, featureImportance = decisionTree(xTrain, xTest, tTrain, tTest, maxIter)
    excutionTime = (time.time() - st)
    msg = 'DecisionTree : ' + str(bestAcc * 100) + '% , ' + str(excutionTime) + ' seconds'
    scoreList.append(msg)
    np.savetxt('dctFeatureImportance.txt', featureImportance)
    

    print("Random Forest running ......")
    st = time.time()
    bestAcc1, bestRfc, trainedScore, featureImportance= randomForest(xTrain, xTest, tTrain, tTest, maxIter)
    excutionTime = (time.time() - st)
    msg = 'RandomForest : ' + str(bestAcc1 * 100) + '% , ' + str(excutionTime) + ' seconds'
    scoreList.append(msg)
    np.savetxt('rfcFeatureImportance.txt', featureImportance)


    print("Naive Bayes running ......")
    st = time.time()
    bestAcc2, bestGnb, trainedScore, featureImportance= naiveBayes(xTrain, xTest, tTrain, tTest, maxIter)
    excutionTime = (time.time() - st)
    msg = 'NaiveBayes : ' + str(bestAcc2 * 100) + '% , ' + str(excutionTime) + ' seconds'
    scoreList.append(msg)
    np.savetxt('gnbFeatureImportance.txt', featureImportance)

    print("Linear Regression running ......")
    st = time.time()
    bestAcc3, accs, featureImportance= LinearRegression.runRegreesion(maxIter)
    excutionTime = (time.time() - st)
    msg = 'LinearRegression : ' + str(bestAcc3 * 100) + '% , ' + str(excutionTime) + ' seconds'
    scoreList.append(msg)
    np.savetxt('lrFeatureImportance.txt', featureImportance)

    print("Quadratic Discriminant Analysis running ......")
    st = time.time()
    bestAcc4, featureImportance= WQDA(xTrain, xTest, tTrain, tTest, maxIter)
    excutionTime = (time.time() - st)
    msg = 'QuadraticDiscriminantAnalysis : ' + str(bestAcc4 * 100) + '% , ' + str(excutionTime) + ' seconds'
    scoreList.append(msg)
    np.savetxt('wqdaFeatureImportance.txt', featureImportance)

    print("\nAlgorithm finished running......")
    print("\nGenerating Report......")
    generateReport.report(scoreList)

    et = (time.time() - tst)
    print("\nReport generated, total excution time is:", str(et))


    

    