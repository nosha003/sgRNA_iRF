#' @references Cliff, Ashley, et al. "A high-performance computing implementation of iterative random forest for the creation of predictive expression networks." Genes 10.12 (2019): 996.
#' @references Walker AM, Cliff A, Romero J, Shah MB, Jones P, Gazolla JG, Jacobson DA, Kainer D. Evaluating the performance of random forest and iterative random forest based methods when applied to gene expression data. Computational and Structural Biotechnology Journal. 2022 Jan 1;20:3372-86. 

import pandas as pd
import numpy as np
import os
import sys
import argparse
import math
import concurrent.futures as cc
import os.path
import sklearn.metrics as mets
import statistics
import time

parser = argparse.ArgumentParser(description="This is a script that is used to run the post processing steps for iRF-LOOP and iRF c
ross-layer, with or without prediction runs. NOTE: ONLY NON-ZERO EDGES WRITTEN TO FILE. FILE MAY NOT REFLECT FULL EDGE LIST.")
parser.add_argument("YNames",help="A single column file where each line is the name of a Y column from the iRF-LOOP or cross-layer 
runs.")
parser.add_argument("RunName",help="The run name used with the --RunName flag in the iRF-Loop or cross-layer runs.",type=str)
parser.add_argument("--Iterations",help="The number of iRF iterations used in the runs.Default=5",type=int,default=5)
parser.add_argument("--Prediction",help="A binary flag to use if the loop or cross-layer runs included kfold prediction runs.",acti
on='store_true')
parser.add_argument("--Kfold",help="Flag to use if Prediction, the number of kfolds that were created during the iRF runs.Default=5
",type=int,default=5)
parser.add_argument("--foldRepeats",help="Flag to use if Prediction, the number of k-fold sets repeated.Default=10",type=int,defaul
t=10)
parser.add_argument("--PredAccuracy",help="The types of prediction accuracy to calculate, comma separated list. MAE is MeanAbsolute
Error/mean, MAEA is MeanAbsoluteError/MeanAbsolute, medianAE is MedianAbsoluteError/Median.Default=MAE, MeanSquaredError=MSE,Coeffi
cient of determination (R squared)=R2",type=str,default='MAE')
parser.add_argument("--varTot",help="The percent cut off of top variance in importance scores to return.Default=55",type=int,defaul
t=95)
parser.add_argument("--numWorkers",help="The number of parallel workers, or threads. default=1.",type=int,default=1)
parser.add_argument("--skipFull",help="A flag to add if you need to skip checking the 'full' run. default=false.",action='store_tru
e',default=False)

args = parser.parse_args()

#feature,args.RunName,currentDir,args.Iterations,args.Prediction,args.Kfold,args.foldRepeats,args.predAccuracy,args.varTot,args.ski
pFull
def featureCalcs(yfeature,runName,currentDir,iterations,prediction,folds,repeats,predAccuracy,varTot,skipFull):
    timeList = []
    startCalcs = time.process_time()
    if not skipFull:
        #Check if time file exists for full irf run
        timeFile = currentDir + '/' + str(yfeature) + '/' + str(runName) + '_' + str(yfeature) + '0.time'
        if not os.path.isfile(timeFile):
            errorFile = currentDir + '/Errors/' + str(yfeature) 
            open(errorFile,'a').close()
            return    
    
        #Check if time file complete for full irf run
        lineCount = 0
        #lastLine = []
        with open(timeFile,'r') as tf:
            for line in tf:
                lineCount = lineCount + 1
                #Full time file will have three more lines than number of iterations
                #if lineCount == (iterations+3):
                #    lastLine.append(line.strip().split(" "))
    
        #Final file count should be 3 + number of iterations, if not then run did not complete
        if lineCount != (iterations+3):
            errorFile = currentDir + '/Errors/' + str(yfeature)
            open(errorFile,'a').close()
            return
    
        #Get importance file from full iRF run
        fullFile = currentDir + '/' + str(yfeature) + '/' + str(runName) + '_' + str(yfeature) + '.importance' + str(iterations-1)

        #Normalize Edges
        data = pd.read_csv(fullFile,sep=" ",index_col=False,header=None,names=['Var','Val'])
        data['NormVal'] = data['Val']/data['Val'].sum(axis=0)
        data['Var'] = data['Var'].map(lambda x: x.strip(":"))

        #Add column with y feature name
        data['YVec'] = yfeature

        #Remove zero edge weight lines
        data = data[data['NormVal'] != 0]

        #Write to file
        outFile = currentDir + '/normalizedEdgeFiles/' + str(yfeature) + '_Normalize.txt'
        data.to_csv(outFile,sep='\t',columns=['Var','YVec','NormVal'],index=None,header=None)
    
        #Sort data to get top var values
        #Using code originally written by Angie Walker
        data = data.sort_values(by=["NormVal"], ascending=False)
        impList = data["NormVal"].tolist()
        featureList = data["Var"].tolist()
        featuresToKeep = []
        currentSum = 0
        count = 0
        for i in impList:
            if currentSum < varTot/100:
                currentSum += i
                featuresToKeep.append(featureList[count])
            count += 1
        topVar = data[data["Var"].isin(featuresToKeep)]
        topVar = topVar.reset_index()
        topVar = topVar.drop(columns=["index"])
        topVarFile = currentDir + '/topVarEdges/' + str(yfeature) + '_top' + str(varTot) + '.txt'
        topVar.to_csv(topVarFile,sep='\t',columns=['Var','YVec','NormVal'],index=None,header=None)




    startPred = time.process_time()
    timeList.append(("fullFile: ", startPred-startCalcs))
    predTypes = predAccuracy.strip().split(',')

    #If results for kfold prediction runs
    if prediction:
        try:
            newDir = currentDir + '/' + str(yfeature) + '/foldRuns/results'
            os.mkdir(newDir)
        except OSError:
            print("Creation of the directory %s failed" % newDir)
       
        predTypes = predAccuracy.strip().split(',')
        #Repeat for each predType
        for predType in predTypes:

            setAccuracy = []
            setAccuracyTup = []
            impScores = {}
        
            #Calculate accuracy for each fold and averaged/median importance scores
            for repeat in range(repeats):
                for fold in range(folds):
                    #print("Feature: " + str(yfeature) + " Fold: " + str(repeat) + " Set: " + str(fold) + " PredType: " + str(predT
ype),flush=True)
                    predFold = time.process_time()
                    trueYFile = currentDir + '/' + str(yfeature) + '/foldRuns/fold' + str(repeat) + '/Runs/Set' + str(fold) + '/set
' + str(fold) + '_Y_test_noSampleIDs.txt'
                    trueY = []
                    lineCount = 0
                    with open(trueYFile,'r') as f:
                        for line in f:
                            if lineCount != 0:
                                trueY.append(line.strip()) 
                            lineCount = lineCount + 1
                    trueY = np.asarray(trueY).astype(np.float64)
                
                    #predYFile = currentDir + '/' + str(yfeature) + '/foldRuns/fold' + str(repeat) + '/Runs/Set' + str(fold) + '/' 
+ str(runName) + '_Set' + str(yfeature) + '_test.prediction'
                    predYFile = currentDir + '/' + str(yfeature) + '/foldRuns/fold' + str(repeat) + '/Runs/Set' + str(fold) + '/' +
 str(runName) + '_Set' + str(fold) + '_test.prediction'
                    
                    trainYFile = currentDir + '/' + str(yfeature) + '/foldRuns/fold' + str(repeat) + '/Runs/Set' + str(fold) + '/' 
+ str(runName) + '_' + str(yfeature) + '0.time'
                    #Check if train file exists
                    #print("Check train file exists. Feature: " + str(yfeature) + " Fold: " + str(repeat) + " Set: " + str(fold) + 
" PredType: " + str(predType),flush=True)
                    if not os.path.isfile(trainYFile):
                        errorFile = currentDir + '/Errors/' + str(yfeature) + '_repeat' +str(repeat) + '_fold' + str(fold) + '_trai
n'
                        open(errorFile,'a').close()
                        #print('No train file ' + str(errorFile),flush=True)
                        continue
                    #print("trian file exists: Feature: " + str(yfeature) + " Fold: " + str(repeat) + " Set: " + str(fold) + " Pred
Type: " + str(predType),flush=True)
                    
                    #Check if train time file is complete
                    #print("Check train file lines. Feature: " + str(yfeature) + " Fold: " + str(repeat) + " Set: " + str(fold) + "
 PredType: " + str(predType),flush=True)
                    lineCountTrain = 0
                    with open(trainYFile,'r') as tf:
                        for line in tf:
                            lineCountTrain = lineCountTrain + 1
                            #Full time file will have three more lines than number of iterations
                            #if lineCount == (iterations+3):
                            #    lastLine.append(line.strip().split(" "))

                        #Final file count should be 3 + number of iterations, if not then run did not complete
                    if lineCountTrain != (iterations+3):
                        errorFile = currentDir + '/Errors/' + str(yfeature) + '_repeat' +str(repeat) + '_fold' + str(fold) + '_trai
n'
                        open(errorFile,'a').close()
                        #print('Not enough train lines: ' + str(lineCountTrain) + " " + str(errorFile),flush=True)
                        continue
                    #print("Trian file lines good. Feature: " + str(yfeature) + " Fold: " + str(repeat) + " Set: " + str(fold) + " 
PredType: " + str(predType),flush=True)
                    
                    #Check if pred file exists
                    #print("Check pred file exists. Feature: " + str(yfeature) + " Fold: " + str(repeat) + " Set: " + str(fold) + "
 PredType: " + str(predType),flush=True)
                    if not os.path.isfile(predYFile):
                        errorFile = currentDir + '/Errors/' + str(yfeature) + '_repeat' +str(repeat) + '_fold' + str(fold) + '_test
'
                        open(errorFile,'a').close()
                        #print("No pred file " + str(errorFile),flush=True)
                        continue
                    #print("Pred file good. Feature: " + str(yfeature) + " Fold: " + str(repeat) + " Set: " + str(fold) + " PredTyp
e: " + str(predType),flush=True)

                    predY = []
                    lineCount = 0
                    with open(predYFile,'r') as f:
                        for line in f:
                            if lineCount != 0:
                                predY.append(line.strip())
                            lineCount = lineCount + 1
                    predY = np.asarray(predY).astype(np.float64)
                    if predType == 'MAE': 
                        error = mets.mean_absolute_error(trueY,predY)
                        mean = np.mean(trueY)
                        setAccuracyTup.append((error,mean,error/mean))
                        setAccuracy.append(error/mean)
                
                    elif predType == "MAEA":
                        error = mets.mean_absolute_error(trueY,predY)
                        mean = np.mean(np.absolute(trueY))
                        setAccuracyTup.append((error,mean,error/mean))
                        setAccuracy.append(error/mean)

                    elif predType == "medianAE":
                        error = mets.median_absolute_error(trueY,predY)
                        median = np.median(trueY)
                        setAccuracyTup.append((error,median,error/median))
                        setAccuracy.append(error/median)

                    elif predType == "MSE":
                        error = mets.mean_squared_error(trueY,predY)
                        setAccuracy.append(error)                  

                    elif predType == "R2":
                        error = mets.r2_score(trueY,predY)
                        setAccuracy.append(error)


            #Write MAE scores to files
            if predType == "MAE":
                #Check if all accuracies were calculated
                if len(setAccuracy) == repeats*folds:
                    outFile = currentDir + '/' + str(yfeature) + '/foldRuns/results/MAE_foldResults.txt'
                    with open(outFile,'w') as f:
                        f.write('MAE\tMean\tMAE/Mean\n')
                        for i in range(len(setAccuracyTup)):
                            f.write(str(setAccuracyTup[i][0])+'\t'+str(setAccuracyTup[i][1])+'\t'+str(setAccuracyTup[i][2])+'\n')
        
            elif predType == "MAEA":
                if len(setAccuracy) == repeats*folds:
                    outFile = currentDir + '/' + str(yfeature) + '/foldRuns/results/MAEA_foldResults.txt'
                    with open(outFile,'w') as f:
                        f.write('MAE\tMeanAbsolute\tMAE/MeanAbsolute\n')
                        for i in range(len(setAccuracyTup)):
                            f.write(str(setAccuracyTup[i][0])+'\t'+str(setAccuracyTup[i][1])+'\t'+str(setAccuracyTup[i][2])+'\n')

            elif predType == "medianAE":
                if len(setAccuracy) == repeats*folds:
                    outFile = currentDir + '/' + str(yfeature) + '/foldRuns/results/medianAE_foldResults.txt'
                    with open(outFile,'w') as f:
                        f.write("medianAE\tMedian\tmedianAE/Median\n")
                        for i in range(len(setAccuracyTup)):
                            f.write(str(setAccuracyTup[i][0])+'\t'+str(setAccuracyTup[i][1])+'\t'+str(setAccuracyTup[i][2])+'\n')

            elif predType == "MSE":
                if len(setAccuracy) == repeats*folds:
                    outFile = currentDir + '/' + str(yfeature) + '/foldRuns/results/MSE_foldResults.txt'
                    with open(outFile,'w') as f:
                        f.write('MSE\n')
                        for i in range(len(setAccuracy)):
                            f.write(str(setAccuracy[i])+'\n')

            elif predType == "R2":
                if len(setAccuracy) == repeats*folds:
                    outFile = currentDir + '/' + str(yfeature) + '/foldRuns/results/R2_foldResults.txt'
                    with open(outFile,'w') as f:
                        f.write('R2\n')
                        for i in range(len(setAccuracy)):
                            f.write(str(setAccuracy[i])+'\n')

            if predType == "MAE":
                outFile = currentDir + '/Accuracy/MAETotal_' + str(yfeature) + '.txt'
            elif predType == "MAEA":    
                outFile = currentDir + '/Accuracy/MAEATotal_' + str(yfeature) + '.txt'
            elif predType == "medianAE":
                outFile = currentDir + '/Accuracy/medianAETotal_' + str(yfeature) + '.txt'
            elif predType == "MSE":
                outFile = currentDir + '/Accuracy/MSETotal_' + str(yfeature) + '.txt'
            elif predType == "R2":
                outFile = currentDir + '/Accuracy/R2Total_' + str(yfeature) + '.txt'
            setLen = len(setAccuracy)
            repFolds = repeats*folds
            if len(setAccuracy) == repeats*folds:
                with open(outFile,'w') as f:
                    f.write(str(yfeature)+'\t'+str(statistics.mean(setAccuracy))+'\n')

        maeEnd = time.process_time()
        
        #Normalize imps
        for repeat in range(repeats):
            for fold in range(folds):
                #Normalize imp file
                #These code snippets are based on code written by Angie Walker
                impFile = currentDir + '/' + str(yfeature) + '/foldRuns/fold' + str(repeat) + '/Runs/Set' + str(fold) + '/' + str(r
unName) + '_' + str(yfeature)+ '.importance' + str(iterations-1)
                #Check if imp file exists
                if not os.path.isfile(impFile):
                    errorFile = currentDir + '/Errors/' + str(yfeature) + '_repeat' + str(repeat) + '_fold' + str(fold) + '_imp'
                    open(errorFile,'a').close()
                    continue
                imp = pd.read_csv(impFile,sep=":",names=["Feature","Importance"],dtype={"Feature":str,"Importance":float})
                imp["normImp"] = imp["Importance"]/imp["Importance"].sum(axis=0)
                #Add imp scores to dict
                for index,row in imp.iterrows():
                    if row["Feature"] not in impScores.keys():
                        impScores[row["Feature"]] = [row["normImp"]]
                    else:
                        impScores[row["Feature"]].append(row["normImp"])
                predFoldEnd = time.process_time()
                timeList.append(("fold time: ", predFoldEnd-predFold))

        #Calculate median and mean importance scores, write to file
        outFile = currentDir + '/' + str(yfeature) + '/foldRuns/results/importanceScores.txt'
        if impScores:
            firstKey = next(iter(impScores))
            if len(impScores[firstKey]) == repeats*folds:
                with open(outFile,'w') as f:
                    f.write("Feature\tMeanImportance\tMedianImportance\n")
                    for feature in impScores.keys():
                        medianScore = statistics.median(impScores[feature])
                        meanScore = statistics.mean(impScores[feature])
                        f.write(str(feature)+'\t'+str(meanScore)+'\t'+str(medianScore)+'\n')
    endPreds = time.process_time()
    if args.Prediction:
        timeList.append(("imp file time: ", endPreds-maeEnd))
    timeList.append(("pred time: ",endPreds-startPred))
    timeList.append(("full feature time: ", endPreds-startCalcs))
    #return timeList
    #return (setLen,repFolds)
    return timeList




        
startAll = time.process_time()
currentDir = os.getcwd()

#Make dir to store normalized, full iRF run results
try:
    newDir = currentDir + '/normalizedEdgeFiles'
    os.mkdir(newDir)
except OSError:
    print("Creation of the directory %s failed" % newDir)

#Make dir to list features that had errors in run completion
try:
    newDir = currentDir + '/Errors'
    os.mkdir(newDir)
except OSError:
    print("Creation of the directory %s failed" % newDir)

#make dir for top variance edges
try:
    newDir = currentDir + '/topVarEdges'
    os.mkdir(newDir)
except OSError:
    print("Creation of the directory %s failed" % newDir)

if args.Prediction:
    try:
        newDir = currentDir + '/Accuracy'
        os.mkdir(newDir)
    except OSError:
        print("Creation of the directory %s failed" % newDir)

#Get list of feature names used in iRF runs
featureFile = args.YNames
features = []
with open(featureFile,'r') as f:
    for line in f:
        features.append(line.strip())

#Create threads to run processing for each feature
runNameList = [args.RunName]*len(features)
currentDirList = [currentDir]*len(features)
iterationsList = [args.Iterations]*len(features)
predictionList = [args.Prediction]*len(features)
kFoldList = [args.Kfold]*len(features)
foldRepeatsList = [args.foldRepeats]*len(features)
predAccuracyList = [args.PredAccuracy]*len(features)
varTotList = [args.varTot]*len(features)
skipFullList = [args.skipFull]*len(features)
#feature,args.RunName,currentDir,args.Iterations,args.Prediction,args.Kfold,args.foldRepeats,args.predAccuracy,args.varTot,args.ski
pFull
with cc.ProcessPoolExecutor(max_workers=args.numWorkers) as executor:
    for feature,result in zip(features,executor.map(featureCalcs,features, runNameList, currentDirList, iterationsList, predictionL
ist, kFoldList, foldRepeatsList, predAccuracyList,varTotList,skipFullList)):
        print(feature,result)


endAll = time.process_time()
print("Full script time: ", endAll-startAll)

