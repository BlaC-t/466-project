import numpy as np
def topFeatures(name, feature, importances, lineBlocker):
    msg = ['\n\n']
    msg.append("Algorithm name: " + name + '\n' + lineBlocker + '\n')
    w = np.absolute(importances).tolist()
    w = [var for var in w if var != 0]

    if len(w) < 10:
        for i in range(len(w)):
            bestW = max(w)
            fmt = "{:<3} {:<15} {:<15}\n"
            msg.append(fmt.format(i+1, feature[w.index(bestW)], str(bestW)))
            feature.remove(feature[w.index(bestW)])
            w.remove(bestW)
    else:
        for j in range(10):
            bestW = max(w)
            fmt = "{:<3} {:<15} {:<15}\n"
            msg.append(fmt.format(j+1, feature[w.index(bestW)], str(bestW)))
            feature.remove(feature[w.index(bestW)])
            w.remove(bestW)
    
    file = open('Report.txt', 'a')
    file.writelines(msg)
    file.close()
    
    g = msg[2:]
    return g

def report(scores):

    algDict = {}
    for eachAlg in scores:
        alg = eachAlg.split()[0]
        acc = eachAlg.split()[2]
        time = eachAlg.split()[4]
        algDict[alg] = (acc, time)

    finalReport = open('Report.txt', 'w')

    introMsg = "Alogrithm name, its accuracy and its excution time\n"
    lineBlocker = "=" * 50
    finalReport.write(introMsg)
    finalReport.write(lineBlocker)
    for each in algDict:
        algMsg = '\n' + each + ': ' + str(algDict[each])
        finalReport.write(algMsg)
    finalReport.close()

    f = open('features.txt', 'r').readlines()
    features = []
    for i in f:
        features.append(i.replace('\n', ''))

    featureList = []

    features1 = features.copy()
    importances= np.loadtxt('dctFeatureImportance.txt')
    freq = topFeatures('Decision Tree', features1, importances, lineBlocker)
    featureList.append(freq)

    features2 = features.copy()
    importances= np.loadtxt('rfcFeatureImportance.txt')
    freq = topFeatures('Random Forest', features2, importances, lineBlocker)
    featureList.append(freq)
    
    features3 = features.copy()
    importances= np.loadtxt('gnbFeatureImportance.txt')
    freq = topFeatures('Naive Bayes', features3, importances, lineBlocker)
    featureList.append(freq)
    
    features4 = features.copy()
    importances= np.loadtxt('lrFeatureImportance.txt')
    freq = topFeatures('Linear Regression', features4, importances, lineBlocker)
    featureList.append(freq)

    features5 = features.copy()
    importances= np.loadtxt('wqdaFeatureImportance.txt')
    freq = topFeatures('Quadratic Discriminant Analysis', features5, importances, lineBlocker)
    featureList.append(freq)

    # print frequency
    tfl = []
    for i in range(len(featureList)):
        fl = []
        for j in featureList[i]:
            fl.append(j.split()[1])
        tfl.append(fl)

    fq = {}
    for i in tfl:
        for j in i:
            if j in fq:
                fq[j] += 1
            else:
                fq[j] = 1
    sfq = dict(sorted(fq.items(), key=lambda x:x[1], reverse=True))

    feRe = open('Report.txt', 'a')
    feRe.write("\n\nFeature Mapping\n")
    feRe.write(lineBlocker)
    feRe.write('\n')
    for each in sfq:
        msg = each + ': ' + str(sfq[each]) + '\n'
        feRe.write(msg)
    feRe.close()

    print(sfq)
    
    
    # print mapping
    mapping = open('mapping.txt', 'r')
    mappingContent = mapping.read()
    mapping.close()

    mapRe = open('Report.txt', 'a')

    mapRe.write("\n\nFeature Mapping\n")
    mapRe.write(lineBlocker)
    mapRe.write('\n')
    mapRe.write(mappingContent)
    mapRe.close()
    

