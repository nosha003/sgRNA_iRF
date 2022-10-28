import pandas as pd
import numpy as np
import math
import sys

if(len(sys.argv)<2):
    print("error needs more args")
    exit()



path = sys.argv[1]
splitFeature1 = 'vacs_score'
pathLines = []
sampLines = []

with open(path,'r') as pathFile:
    for line in pathFile:
        depVar = 0.0
        numSamples = 0.0
        features = []
        splitVals = []
        direction = []
        splitFeatureList1 = []
        newLine = line.rstrip(' \n').split(" ")
        #print(newLine)
        numSamples = (newLine[0])
        depVar = (newLine[1])
        splitFeatureList = []
        for i in range(2,len(newLine),3):
            features.append(newLine[i])
            splitVals.append(newLine[i+1])
            direction.append(newLine[i+2])
        featuresNP = np.array(features)
        pathLines.append(features)
        sampLines.append(int(numSamples))
        splitFeatureIndices1 = np.where(featuresNP == splitFeature1)[0]
        #print(features)
        #print (splitFeatureIndices1)
        if splitFeatureIndices1.size > 0:
            for i in range(len(splitFeatureIndices1)):
                #For each location in the splitFeatureList, get the corresponding info of the other lists
                splitFeatureList1.append([depVar, splitFeatureIndices1[i], float(splitVals[splitFeatureIndices1[i]]), int(direction
[splitFeatureIndices1[i]])])
        #print(splitFeatureList1)
        if len(splitFeatureList1) > 0:
            #If there is only one location where the specified feature is split upon, check if that split val is above or below the
 cut off value
            if len(splitFeatureList1) == 1:
                if splitFeatureList1[0][3] == 0:
                    featuresLess.append([(splitFeaturesList[0][2]),newLine])
                if splitFeatureList1[0][3] == 1:
                    featuresGreater.append(newLine)
            #If there are more than one location in the path where the specified features is split upon, then check the first and l
ast location to create a range
            else:
                minFeature = math.inf
                maxFeature = -math.inf
                for i in range(len(splitFeatureList1)):
                    #check values and directions to create range
                    #maxVal and 0 direction will set top of range
                    #minVal and 1 direction will set bottom of range
                    #if minFeature or maxFeature is still inf at end, then only one sided range is created
                    if splitFeatureList1[i][2] >= maxFeature and splitFeatureList1[i][3] == 0:
                        maxFeature = splitFeatureList1[i][2]
                    if splitFeatureList1[i][2] <= minFeature and splitFeatureList1[i][3] == 1:
                        minFeature = splitFeatureList1[i][3]
                #If no minimum was set, use 
                #if minFeature == math.inf:
                #    featuresLess.append(newLine)
                #if maxFeature == -math.inf:
                #    featuresGreater.append(newLine)
        #print()         
         

flat_paths = [item for sublist in pathLines for item in sublist]
featureSet = list(set(flat_paths))
feature_to_number = dict(zip(featureSet, list(range(len(featureSet)))))

keyout=path+".key.out"
print("Printing key")
with open(keyout, 'w') as f:
    for key, val in feature_to_number.items():
        f.write("%s, " % key)
        f.write("%s" % val)
        f.write("\n")

pathout=path+".int.out"
print("printing path file")
with open(pathout, 'w') as f:
    count=0
    for item in pathLines:
        f.write("%s " % sampLines[count])
        count+=1
        for thing in item:
           f.write("%s " % feature_to_number[thing])
        f.write("\n")

print("counting features")
fullset=[]
wordcount=dict()
count=0
for ylist in pathLines:
	fullset.append(len(ylist))
	for word in list(set(ylist)):
		if word in wordcount:
			wordcount[word] = wordcount[word] + sampLines[count]
		else:
			wordcount[word] = sampLines[count]
	count+=1

infoOut=path+".info.out"
with open(infoOut, 'w') as f:
	f.write("number_of_paths: %s\n" % str(len(fullset)))
	f.write("max: %s\n" % str(max(fullset)))
	f.write("average: %s\n" % str(sum(fullset)/len(fullset)))

keyout=path+".wordcount.out"
print("Printing wordcount")
with open(keyout, 'w') as f:
    sampSum=np.sum(sampLines)
    for key, val in wordcount.items():
        f.write("%s, " % key)
        f.write("%s" % str(int(val)/sampSum))
        f.write("\n")  
