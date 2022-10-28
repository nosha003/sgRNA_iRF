import numpy as np
import pandas as pd
import concurrent.futures as cc
#from multiprocessing import shared_memory
import sys
import time
import functools
from RIThelper import *

if(len(sys.argv)<2):
    print("error needs more args")
    exit()

start_time=time.time()
ImpFileName=sys.argv[1]
name=ImpFileName.split('.importance')[0]
yVec=sys.argv[2]

threaded=False
if (len(sys.argv) == 4):
    threaded=True
    
paths=name+'.paths'
paths_key_pd = pd.read_csv(name+".paths.key.out",header=None,dtype={'0':np.int64,'1':str})
paths_key = dict(zip(paths_key_pd[1].values.tolist(),paths_key_pd[0].values.tolist()))

print("read key file: "+str(time.time()-start_time),flush=True)
read_key_time=time.time()
pathInfo,pathLoc=decompForest(paths)

print("process paths: " + str(time.time()-read_key_time),flush=True)
Paths_time=time.time()

ImpDF=pd.read_csv(ImpFileName,index_col=None,sep=':',header=None,names=['Feature','Edge'])
print("read importance file",flush=True)

#temp_time=time.time()

features=ImpDF['Feature'].values.tolist()
featureNumbers=list(range(len(features)))

FeatureEffect=[]
Samples=[]
maxEffect=[]
minEffect=[]
linearity=[]

#print("prep paths: " + str(time.time()-Paths_time),flush=True)

#tempImpFunction = lambda feature: ImportanceInfo(pathLoc,feature)

#shared_pathLoc=shared_memory.

if threaded:
    FeatureEffect=[0]*len(features)
    Samples=[0]*len(features)
    linearity=[0]*len(features)
    pathLocCopied=[pathLoc]*len(features)

    with cc.ProcessPoolExecutor(max_workers=None) as executor:
        for num,result in zip(featureNumbers,executor.map(ImportanceInfo,pathLocCopied,features)):
            Samples[num]=result[1]
            FeatureEffect[num]=result[2]
            linearity[num]=result[3]
else:        
    for x in features:
        #print(ImpDF['Feature'].iloc[x])
        results=ImportanceInfo(pathLoc,str(x))
        #print(ImpDF['Feature'].iloc[x], " time: ", str(time.time()-temp_time),flush=True)
        #temp_time=time.time()
        Samples.append(results[1])
        FeatureEffect.append(results[2])
        linearity.append(results[3])


print("process importance: " + str(time.time()-Paths_time),flush=True)


ImpDF['Samples']=Samples
ImpDF['FeatureEffect']=FeatureEffect
ImpDF['Linearity']=linearity

ImpDF['NormEdge'] = ImpDF['Edge']/ImpDF['Edge'].sum(axis=0)
ImpDF['YVec']= yVec

ImpDF = ImpDF[ImpDF['NormEdge'] > 0]
ImpDF = ImpDF.dropna()

ImpDF[['Feature','YVec','NormEdge','FeatureEffect','Samples','Linearity']].to_csv(ImpFileName+'.effect',sep='\t',index=None)

paths_count_pd = pd.read_csv(name+".paths.wordcount.out",header=None)
paths_count=dict()
for x in range(len(paths_count_pd[0].values.tolist())):
    paths_count[str(paths_count_pd[0].values.tolist()[x])] = paths_count_pd[1].values.tolist()[x]

print("read wordcount file: "+str(time.time()-Paths_time) ,flush=True)
read_wordcount_time=time.time()

pathsRit=[]
with open(name+'.rit') as f:
    lines = f.readlines()
    for line in lines:
        temp=[]
        splitline=line.strip().split('\t')
        temp.append(splitline[0])
        temp2=[]
        for x in splitline[1].split(','):
            temp2.append(paths_key[int(x)])
        temp.append(temp2)
        pathsRit.append(temp)

print("read RIT file: " + str(time.time()-read_wordcount_time),flush=True)
read_RIT_time=time.time()

for x in range(len(pathsRit)):
    vals=pathsRit[x]
    #print(vals)
    tempOut=1
    featList=[]
    for feature in vals[1]:
        feature=str(feature)
        featList.append(feature)
        tempOut*=paths_count[feature]
    vals[0]=float(vals[0])
    vals.append(vals[0]-tempOut)
    vals+=ConImpSet(pathInfo,pathLoc,featList)
    vals[3]=(vals[3])/(ImpDF['Edge'].sum(axis=0))

print("print rit adjusted file: "+str(time.time()-read_RIT_time),flush=True)
RIT_adj_time=time.time()


pathout=name+".rit.adj"
with open(pathout, 'w') as f:
    f.write("RIT\tRIT_adjusted\tSet_Importance\tNumSamples\tConditionalImportance\tConditionalImportanceList\tMaxEffect\tMinEffect\
tFeatureSet\n")
    for item in pathsRit:
        f.write("%s\t" % item[0])
        f.write("%s\t" % item[2])
        f.write("%s\t" % item[3])
        f.write("%s\t" % item[4])
        f.write("%s\t" % item[5])
        f.write("%s\t" % item[6])
        f.write("%s\t" % item[7])
        f.write("%s\t" % item[8])
        for thing in item[1]:
            f.write("%s," % thing)
        f.write("\n")

print("print rit adjusted edge file: "+str(time.time()-RIT_adj_time),flush=True)

edgeout=name+".rit.edge"
with open(edgeout, 'w') as f:
    f.write("Source\tSink\tRIT\tRIT_adjusted\tSet_Importance\tNumSamples\tAverageEffect\tMaxEffect\tMinEffect\tLinearity\tIsSetEdge
\n")
    for item in pathsRit:
        for thing in item[1]:
            f.write("%s\t" % thing)
            for thing1 in item[1]:
                f.write("%s_" % thing1)
            f.write("\t1\t1\t1\t1\t1\t1\t1\t1\t0\n")
        for thing in item[1]:
            f.write("%s_" % thing)
        f.write("\t%s\t" % yVec)
        f.write("%s\t" % item[0])
        f.write("%s\t" % item[2])
        f.write("%s\t" % item[3])
        f.write("%s\t" % item[4])
        f.write("%s\t" % item[5])
        f.write("%s\t" % item[6])
        f.write("%s\t" % item[7])
        f.write("%s\t1\n" % item[8])

print("total time in ritEval.py: " + str(time.time()-start_time))
