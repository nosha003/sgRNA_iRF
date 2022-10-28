#' @references Cliff, Ashley, et al. "A high-performance computing implementation of iterative random forest for the creation of predictive expression networks." Genes 10.12 (2019): 996.
#' @references Walker AM, Cliff A, Romero J, Shah MB, Jones P, Gazolla JG, Jacobson DA, Kainer D. Evaluating the performance of random forest and iterative random forest based methods when applied to gene expression data. Computational and Structural Biotechnology Journal. 2022 Jan 1;20:3372-86.

#Last edited 1/26/2020 - Added targetNodeSize parameter
#Last edited 3/4/2020 - Changed a range() var to int
#Last edited 4/21/2020 - Brut version updated to use 'MtryType' flag (required when summit code is updated - git master includes it
)
#Last edited 5/18/2020 - Created CrossLayer version
#Last edited 5/27/2020 - Moved some crosslayer commands to with in 'not ReRun' set
#Last edited 8/10/2020 - Added kfold prediction steps
#Last edited 10/28/2020 - Added prediction flag wrapper around kfold process
#Last edited 09/26/2022 - Bug fix in labeling

import argparse
import csv
import sys
import os
import math
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedKFold
import random
import copy
import time
import concurrent.futures as cc
from collections import Counter, defaultdict
from sklearn.model_selection._split import _BaseKFold, _RepeatedSplits
from sklearn.utils.validation import check_random_state


parser = argparse.ArgumentParser(description="This is a script that is used to build the submit scripts to run iRF-LOOP on a set of
 y vectors. The result of running this program is a submit script, or a set of submit scripts, and a 'SubmitAll' script. Each submi
t script contains a number of iRF executables with associated parameters. The 'SubmitAll' script contains the 'submit' command for
that system and the name of each submit script, so that if there are many submit scripts, to submit them all to the queue at once,
simply execute 'bash SubmitAll_<RunName>.sh'. This script will also subset the X matrix and Y vecs to the intersect of samples, and
 will create Y Vec files for each Y vector, in each independent run directory.")
parser.add_argument("DataFile", help="The full path to the data file (X matrix) to be used for iRF. Sample ID column should be the
first column. May contain duplicate ID names. Advised that this file be in the directory the jobs will run from.")
parser.add_argument("YFile",help="The full path to the file containing the Y vectors. Sample ID column should be the first column -
 naming scheme same as X matrix. May NOT contain duplicate ID names.  Advised that this file be in the directory that jobs will run
 from.")
parser.add_argument("--System",help="Compute system. Options are: Summit or Brut",default='Summit')
parser.add_argument("--NodesPer", type=int, help="The number of nodes per each independent iRF model. Used when System = Summit.",
default=5)
parser.add_argument("--TotalNodes", type=int, help="The total number of nodes for an allocation. Determines how many iRF models can
 run simultaneously. NodesPer var must evenly divide TotalNodes. Used when System = Summit.", default=100)
parser.add_argument("--CoresPer", type=int,help="The number of cores to use for each iRF model. Used when System = Brut.",default=1
0)
parser.add_argument("--TotalCores",type=int,help="The total number of cores for an allocation. Determines how many iRF models can r
un simultaneously. It is preferable that this value is an increment of 32, as allocations are in 32 core chunks. CoresPer var must
evenly divide TotalCores. Used when System = Brut.",default=32)
parser.add_argument("--RunTime",type=int,help="Estimated run time for ONE iRF model, in minutes.",default=5)
parser.add_argument("--Account",help="Allocation account name to be used in the submit script",default='SYB105')
parser.add_argument("--NumTrees",type=int,help="The number of trees for one iRF model",default=1000)
parser.add_argument("--NumIterations",type=int,help="The number of iterations for one iRF model",default=5)
parser.add_argument("--RunName",help="The name for the run, will be follewed by the script number.",default="iRFRun")
parser.add_argument("--ReRun",help="Set if doing a rerun or partial run",action='store_true')
parser.add_argument("--ReRunFile", help="The full path to the list (one feature per line) of reRuns.")
parser.add_argument("--targetNodeSize",type=int,help="Maximum node size for leaf",default=5)
parser.add_argument("--bypass",help="Bypass the x and y overlap steps. Pre-overlapped files must be in the form <DataFile>_overlap_
noSampleIDs.txt and <YFile>_overlap_noSampleIDs.txt",action='store_true',default=False)
parser.add_argument("--Prediction",help="Boolean flag to use if doing kfold prediction",action='store_true',default=False)
parser.add_argument("--Kfold",help='Number of k-folds to create.Default=5',type=int,default=5)
parser.add_argument("--foldType",help="Type of kfold to use, groupFold or stratifiedFold. Default is groupFold.",type=str,default='
groupFold')
parser.add_argument("--foldSets",help="Number oof k-fold sets to create.Default = 10.",type=int,default=10)
parser.add_argument("--groupFile",help="File with group ID for each sample in X matrix. One ID per line, matching sample order in m
atrix. If flag is not set, groups are assumed to be sample IDs from the X matrix file.",type=str,default="NAN")
parser.add_argument("--sampleSize",help="Max number of each sample group to use in a run.Default=5.",type=int,default=5)
parser.add_argument("--alwayssplitvars",help="Feature names to include at every split",type=str,default="NAN")

startTime = time.process_time()

args = parser.parse_args()



class StratifiedGroupKFold(_BaseKFold):
    """Stratified K-Folds iterator variant with non-overlapping groups.

    This cross-validation object is a variation of StratifiedKFold that returns
    stratified folds with non-overlapping groups. The folds are made by
    preserving the percentage of samples for each class.

    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).

    The difference between GroupKFold and StratifiedGroupKFold is that
    the former attempts to create balanced folds such that the number of
    distinct groups is approximately the same in each fold, whereas
    StratifiedGroupKFold attempts to create folds which preserve the
    percentage of samples for each class.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.

    random_state : int or RandomState instance, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedGroupKFold
    >>> X = np.ones((17, 2))
    >>> y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> groups = np.array([1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8])
    >>> cv = StratifiedGroupKFold(n_splits=3)
    >>> for train_idxs, test_idxs in cv.split(X, y, groups):
    ...     print("TRAIN:", groups[train_idxs])
    ...     print("      ", y[train_idxs])
    ...     print(" TEST:", groups[test_idxs])
    ...     print("      ", y[test_idxs])
    TRAIN: [2 2 4 5 5 5 5 6 6 7]
           [1 1 1 0 0 0 0 0 0 0]
     TEST: [1 1 3 3 3 8 8]
           [0 0 1 1 1 0 0]
    TRAIN: [1 1 3 3 3 4 5 5 5 5 8 8]
           [0 0 1 1 1 1 0 0 0 0 0 0]
     TEST: [2 2 6 6 7]
           [1 1 0 0 0]
    TRAIN: [1 1 2 2 3 3 3 6 6 7 8 8]
           [0 0 1 1 1 1 1 0 0 0 0 0]
     TEST: [4 5 5 5 5]
           [1 0 0 0 0]

    See also
    --------
    StratifiedKFold: Takes class information into account to build folds which
        retain class distributions (for binary or multiclass classification
        tasks).

    GroupKFold: K-fold iterator variant with non-overlapping groups.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    # Implementation based on this kaggle kernel:
    # https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    def _iter_test_indices(self, X, y, groups):
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, group in zip(y, groups):
            y_counts_per_group[group][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        groups_and_y_counts = list(y_counts_per_group.items())
        rng = check_random_state(self.random_state)
        if self.shuffle:
            rng.shuffle(groups_and_y_counts)

        for group, y_counts in sorted(groups_and_y_counts,
                                      key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                y_counts_per_fold[i] += y_counts
                std_per_label = []
                for label in range(labels_num):
                    std_per_label.append(np.std(
                        [y_counts_per_fold[j][label] / y_distr[label]
                         for j in range(self.n_splits)]))
                y_counts_per_fold[i] -= y_counts
                fold_eval = np.mean(std_per_label)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(group)

        for i in range(self.n_splits):
            test_indices = [idx for idx, group in enumerate(groups)
                            if group in groups_per_fold[i]]
            yield test_indices


class RepeatedStratifiedGroupKFold(_RepeatedSplits):
    """Repeated Stratified K-Fold cross validator.

    Repeats Stratified K-Fold with non-overlapping groups n times with
    different randomization in each repetition.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.

    random_state : int or RandomState instance, default=None
        Controls the generation of the random states for each repetition.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import RepeatedStratifiedGroupKFold
    >>> X = np.ones((17, 2))
    >>> y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> groups = np.array([1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8])
    >>> cv = RepeatedStratifiedGroupKFold(n_splits=2, n_repeats=2,
    ...                                   random_state=36851234)
    >>> for train_index, test_index in cv.split(X, y, groups):
    ...     print("TRAIN:", groups[train_idxs])
    ...     print("      ", y[train_idxs])
    ...     print(" TEST:", groups[test_idxs])
    ...     print("      ", y[test_idxs])
    TRAIN: [2 2 4 5 5 5 5 8 8]
           [1 1 1 0 0 0 0 0 0]
     TEST: [1 1 3 3 3 6 6 7]
           [0 0 1 1 1 0 0 0]
    TRAIN: [1 1 3 3 3 6 6 7]
           [0 0 1 1 1 0 0 0]
     TEST: [2 2 4 5 5 5 5 8 8]
           [1 1 1 0 0 0 0 0 0]
    TRAIN: [3 3 3 4 7 8 8]
           [1 1 1 1 0 0 0]
     TEST: [1 1 2 2 5 5 5 5 6 6]
           [0 0 1 1 0 0 0 0 0 0]
    TRAIN: [1 1 2 2 5 5 5 5 6 6]
           [0 0 1 1 0 0 0 0 0 0]
     TEST: [3 3 3 4 7 8 8]
           [1 1 1 1 0 0 0]

    Notes
    -----
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.

    See also
    --------
    RepeatedStratifiedKFold: Repeats Stratified K-Fold n times.
    """

    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(StratifiedGroupKFold, n_splits=n_splits,
                         n_repeats=n_repeats, random_state=random_state)









#currentDir, featureCount, feature, args.foldSets, dataArray, groups, args.sampleSize, args.Kfold, args.foldType, dataFile, args.Pr
ediction, args.ReRun
def featureFoldRun(currentDir,featureCount,feature,foldSets,dataArray,groups,sampleSize,Kfold,foldType,xFile, prediction,reRun):
    timeList = []
    timeA = time.process_time()
    try:
        newDir = currentDir + '/' + str(feature)
        os.mkdir(newDir)
    except OSError:
        print("Creation of the directory %s failed" % newDir)

    if not args.ReRun:
        #Add Yvec column file to results directory
        cut = "cut -f" + str(featureCount) + " " + str(yFileNoIDs) +" > " + str(newDir) + "/" + str(feature) + "_col.txt"
        print(cut+'\n')
        os.system(cut)

    if prediction:
        try:
            newDir = currentDir + '/' + str(feature) + '/foldRuns'
            os.mkdir(newDir)
        except:
            print("Creation of the directory %s failed" % newDir)

        foldRepeat = 0
        yDataColFile = currentDir + '/' + str(feature) + "/" + str(feature) + "_col.txt"
        lineCount = 0
        yDataCol = []
        with open(yDataColFile,'r') as f:
            for line in f:
                if lineCount != 0:
                    yDataCol.append(float(line.strip()))
                lineCount = lineCount + 1
        nonZerotmp = np.count_nonzero(np.ma.make_mask(yDataCol))
        nonZero = np.ma.make_mask(yDataCol).astype(int)
        timeB = time.process_time()
        if ((nonZerotmp == 0) and (foldType == 'stratifiedFold')) or foldType == 'groupFold':
            print("Fold type: group")
            for foldRepeat in range(foldSets):
                timeC = time.process_time()
                try:
                    newDir = currentDir + '/' + str(feature) +  '/foldRuns/fold' + str(foldRepeat)
                    os.mkdir(newDir)
                except OSError:
                    print("Creation of the directory %s failed" % newDir)

                #Make directory to store job results
                try:
                    newDir = currentDir + '/' + str(feature) + '/foldRuns/fold' + str(foldRepeat) + '/Runs'
                    os.mkdir(newDir)

                except OSError:
                    print("Creation of the directory %s failed" % newDir)

                tmpData = dataArray
                tmpGroups = groups
                tmpIndex = np.array(list(range(len(dataArray))))
                #print(tmpGroups)

                #Count samples in each group
                groupsDir = {}
                for item in tmpGroups:
                    if item not in groupsDir.keys():
                        groupsDir[item] = 1
                    else:
                        groupsDir[item] = groupsDir[item] + 1
                #print(groupsDir)
                #Downselect for high count groups
                highGroups = []
                #print("len: ", str(len(highGroups)))
                for key,value in groupsDir.items():
                    if float(value) > sampleSize:
                        highGroups.append(key)
                        print("HighGroup add: ", str(key))

                #Get indicies for all samples with given group ID, repeat for each high group
                if highGroups:
                    print("Len highGroups: ", len(highGroups))
                    for group in highGroups:
                        #print("Group: ", str(group))
                        indices = []
                        for i, val in enumerate(tmpGroups):
                            if val == group:
                                indices.append(i)
                        removeIndices = np.random.choice(indices, size=(len(indices)-sampleSize),replace=False)
                        tmpData = np.delete(tmpData,removeIndices)
                        tmpGroups = np.delete(tmpGroups, removeIndices)
                        tmpIndex = np.delete(tmpIndex, removeIndices)

                #print("ycol: ", tmpYDataCol)
                #Shuffle groups
                groupData = list(zip(tmpData,tmpGroups,tmpIndex))
                #random.shuffle(groupData)
                #tmpData, tmpGroups, tmpIndex = zip(*groupData)
                #tmpData = np.asarray(tmpData)
                random.shuffle(tmpGroups)
                #print("ycol: ",tmpYDataCol)
                tmpGroups = np.asarray(tmpGroups)
                tmpIndex = np.asarray(tmpIndex)

                kf =GroupKFold(n_splits=Kfold)
                split = kf.split(tmpData,groups=tmpGroups)
                foldCount = 0
                for train_index,test_index in split:
                    timeD = time.process_time()
                    try:
                        newDir = currentDir + '/' + str(feature) + '/foldRuns/fold' + str(foldRepeat) + '/Runs/Set' + str(foldCount
)
                        os.mkdir(newDir)
                    except OSError:
                        print("Creation of the directory %s failed" % newDir)

                    trainIndexFile = currentDir + '/' + str(feature) + '/foldRuns/fold' + str(foldRepeat) + '/Runs/Set' + str(foldC
ount) + '/set' + str(foldCount) + '_trainIndex.txt'
                    #sedCommandTrain = ""
                    with open(trainIndexFile,'w') as f:
                        f.write('1p\n')
                        for index in train_index:
                            f.write(str(tmpIndex[index]+2)+"p\n")
                            #sedCommandTrain += str(index+2)+"p;"

                    sedTrainX = "sed -n -f " + trainIndexFile + " " + dataFileNoIDs + " > " + currentDir + '/' + str(feature) + '/f
oldRuns/fold' + str(foldRepeat) + '/Runs/Set' + str(foldCount) + '/set' + str(foldCount) + '_train_noSampleIDs.txt'
                    #sedTrainX = "sed -n '1p;" + sedCommandTrain + "' " + dataFileNoIDs + " > " + currentDir + '/' + str(feature) +
 '/foldRuns/fold' + str(foldRepeat) + '/Runs/Set' + str(foldCount) + '/set' + str(foldCount) + '_train_noSampleIDs.txt'

                    sedTrainY = "sed -n -f " + trainIndexFile + " " + yDataColFile + " > " + currentDir + '/' + str(feature) + '/fo
ldRuns/fold' + str(foldRepeat) + '/Runs/Set' + str(foldCount) + '/set' + str(foldCount) + '_Y_train_noSampleIDs.txt'
                    #sedTrainY = "sed -n '1p;" + sedCommandTrain + "' " + yDataColFile + " > " + currentDir + '/' + str(feature) +
'/foldRuns/fold' + str(foldRepeat) + '/Runs/Set' + str(foldCount) + '/set' + str(foldCount) + '_Y_train_noSampleIDs.txt'

                    testIndexFile = currentDir + '/' + str(feature) + '/foldRuns/fold' + str(foldRepeat) + '/Runs/Set' + str(foldCo
unt) + '/set' + str(foldCount) + '_testIndex.txt'
                    #sedCommandTest = ""
                    with open(testIndexFile,'w') as f:
                        f.write('1p\n')
                        for index in test_index:
                            f.write(str(tmpIndex[index]+2)+"p\n")
                            #sedCommandTest += str(index+2)+"p;"

                    sedTestX = "sed -n -f " + testIndexFile + " " + dataFileNoIDs + " > " + currentDir + '/' + str(feature) + '/fol
dRuns/fold' + str(foldRepeat) + '/Runs/Set' + str(foldCount) + '/set' + str(foldCount) + '_test_noSampleIDs.txt'
                    #sedTestX = "sed -n '1p;" + sedCommandTest + "' " + dataFileNoIDs + " > " + currentDir + '/' + str(feature) + '
/foldRuns/fold' + str(foldRepeat) + '/Runs/Set' + str(foldCount) + '/set' + str(foldCount) + '_test_noSampleIDs.txt'
                    sedTestY = "sed -n -f " + testIndexFile + " " + yDataColFile + " > " + currentDir + '/' + str(feature) + '/fold
Runs/fold' + str(foldRepeat) + '/Runs/Set' + str(foldCount) + '/set' + str(foldCount) + '_Y_test_noSampleIDs.txt'
                    #sedTestY = "sed -n '1p;" + sedCommandTest + "' " + yDataColFile + " > " + currentDir + '/' + str(feature) + '/
foldRuns/fold' + str(foldRepeat) + '/Runs/Set' + str(foldCount) + '/set' + str(foldCount) + '_Y_test_noSampleIDs.txt'

                    trainSampleFile = currentDir + '/' + str(feature) + '/foldRuns/fold' + str(foldRepeat) + '/Runs/Set' + str(fold
Count) + '/set' + str(foldCount) + '_train_SampleIDs.txt'
                    testSampleFile = currentDir + '/' + str(feature) + '/foldRuns/fold' + str(foldRepeat) + '/Runs/Set' + str(foldC
ount) + '/set' + str(foldCount) + '_test_SampleIDs.txt'

                    os.system(sedTrainX)
                    #print(sedTrainX)
                    os.system(sedTrainY)
                    #print(sedTrainY)
                    os.system(sedTestX)
                    os.system(sedTestY)

                    sortIndexTrain = sorted(tmpIndex[train_index])
                    sortIndexTest = sorted(tmpIndex[test_index])
                    train_samples,test_samples = dataArray[sortIndexTrain],dataArray[sortIndexTest]
                    with open(trainSampleFile,'w') as f:
                        for item in train_samples:
                            f.write(item +'\n')
                    with open(testSampleFile,'w') as f:
                        for item in test_samples:
                            f.write(item+'\n')
                    foldCount = foldCount + 1
                    timeE = time.process_time()
                    timeList.append(timeE-timeD)
                #foldRepeat = foldRepeat + 1
                timeList.append(timeE-timeA)


        elif nonZerotmp != 0 and foldType == 'stratifiedFold':
            timeF = time.process_time()
            kf = RepeatedStratifiedGroupKFold(n_splits=Kfold,n_repeats=foldSets)
            split = kf.split(dataArray,nonZero,groups)
            foldCount = 0
            #foldRepeat = 0
            fullCount = 0
            for train_index,test_index in split:
                timeG = time.process_time()
                if foldCount == 0:
                    try:
                        newDir = currentDir + '/' + str(feature) +  '/foldRuns/fold' + str(foldRepeat)
                        os.mkdir(newDir)
                    except OSError:
                        print("Creation of the directory %s failed" % newDir)

                    #Make directory to store job results
                    try:
                        newDir = currentDir + '/' + str(feature) + '/foldRuns/fold' + str(foldRepeat) + '/Runs'
                        os.mkdir(newDir)

                    except OSError:
                        print("Creation of the directory %s failed" % newDir)

                try:
                    newDir = currentDir + '/' + str(feature) + '/foldRuns/fold' + str(foldRepeat) + '/Runs/Set' + str(foldCount)
                    os.mkdir(newDir)
                except OSError:
                    print("Creation of the directory %s failed" % newDir)

                sedCommandTrain = ""
                for index in train_index:
                    sedCommandTrain += str(index+2)+"p;"
                sedTrainX = "sed -n '1p;" + sedCommandTrain + "' " + dataFileNoIDs + " > " + currentDir + '/' + str(feature) + '/fo
ldRuns/fold' + str(foldRepeat) + '/Runs/Set' + str(foldCount) + '/set' + str(foldCount) + '_train_noSampleIDs.txt'
                sedTrainY = "sed -n '1p;" + sedCommandTrain + "' " + yDataColFile + " > " + currentDir + '/' + str(feature) + '/fol
dRuns/fold' + str(foldRepeat) + '/Runs/Set' + str(foldCount) + '/set' + str(foldCount) + '_Y_train_noSampleIDs.txt'

                sedCommandTest = ""
                for index in test_index:
                    sedCommandTest += str(index+2)+"p;"
                sedTestX = "sed -n '1p;" + sedCommandTest + "' " + dataFileNoIDs + " > " + currentDir + '/' + str(feature) + '/fold
Runs/fold' + str(foldRepeat) + '/Runs/Set' + str(foldCount) + '/set' + str(foldCount) + '_test_noSampleIDs.txt'
                sedTestY = "sed -n '1p;" + sedCommandTest + "' " + yDataColFile + " > " + currentDir + '/' + str(feature) + '/foldR
uns/fold' + str(foldRepeat) + '/Runs/Set' + str(foldCount) + '/set' + str(foldCount) + '_Y_test_noSampleIDs.txt'

                trainSampleFile = currentDir + '/' + str(feature) + '/foldRuns/fold' + str(foldRepeat) + '/Runs/Set' + str(foldCoun
t) + '/set' + str(foldCount) + '_train_SampleIDs.txt'
                testSampleFile = currentDir + '/' + str(feature) + '/foldRuns/fold' + str(foldRepeat) + '/Runs/Set' + str(foldCount
) + '/set' + str(foldCount) + '_test_SampleIDs.txt'

                os.system(sedTrainX)
                os.system(sedTrainY)
                os.system(sedTestX)
                os.system(sedTestY)

                train_samples,test_samples = dataArray[train_index],dataArray[test_index]
                with open(trainSampleFile,'w') as f:
                    for item in train_samples:
                        f.write(item +'\n')
                with open(testSampleFile,'w') as f:
                    for item in test_samples:
                        f.write(item+'\n')

                fullCount = fullCount + 1
                foldCount = fullCount % Kfold
                foldRepeat = int(np.floor(fullCount/Kfold))
                timeH = time.process_time()
                timeList.append(timeH-timeG)
            timeJ = time.process_time()
            timeList.append(timeJ-timeF)
        #print("Fold time: ",timeF-timeB)
    return timeList



dataFile = args.DataFile
yFile = args.YFile

print("DataFile set",flush=True)

dataFileName = os.path.splitext(dataFile)[0]
yFileName = os.path.splitext(yFile)[0]
dataBaseName = os.path.basename(dataFile)
yBaseName = os.path.splitext(os.path.basename(yFile))[0]

with open(dataFile, 'r') as df:
    dataWithHeader = df.readline()
print("dataWithHeader: ", sys.getsizeof(dataWithHeader),flush=True)

with open(yFile, 'r') as df:
    yDataWithHeader = df.readline()

#Determine delimiter
sniffer = csv.Sniffer()
dialect = sniffer.sniff(dataWithHeader)
delimiter = dialect.delimiter
print("Delimiter:" + delimiter + "'\n",flush=True)

#Num columns in data file - includes sample ID col
dataNumCols = len(dataWithHeader.split(delimiter))
print("Data cols: ", dataNumCols,flush=True)

#Num columns in y file - includes sample ID col
yNumCols = len(yDataWithHeader.split(delimiter))
print("yNumCols: ", yNumCols,flush=True)

if (not args.ReRun) and (not args.bypass):

    #Make no-header files
    tail1 = "tail -n+2 " + str(dataFile) + " > " + str(dataFileName) + "_noHeader.txt"
    tail2 = "tail -n+2 " + str(yFile) + " > " + str(yFileName) + "_noHeader.txt"

	#Sort samples of X matrix and YVecs
    sort1 = "sort -k1,1 " + str(dataFileName) + "_noHeader.txt" + " > " + str(dataFileName) + "_noHeader_sort.txt"
    sort2 = "sort -k1,1 " + str(yFileName) + "_noHeader.txt" + " > " + str(yFileName) + "_noHeader_sort.txt"

    #Keeps lines in dataFile that have ID in yFile (doesn't duplicated lines)
    #awk1 = "awk 'NR==FNR{a[$1]++;next} a[$1]>0' " + str(yFile) + "_sort.txt "  + str(dataFile) + "_sort.txt > " + str(dataFile) +
"_overlap_noHeader.txt"
    #Keeps lines in yFile that have ID in dataFile (doesn't duplicate lines)
    #awk2 = "awk 'NR==FNR{a[$1]++;next} a[$1]>0' " + str(dataFile) + "_overlap_noHeader.txt "  + str(yFile) + "_sort.txt > " + str(
yFile) + "_overlap_noHeader.txt_tmp"
    #For each line in dataFile, prints corresponding yData line to file
    #awk3 = "awk 'NR==FNR{a[$1]=$0;next} $1 in a{print a[$1]}' " + str(yFile) + "_overlap_noHeader.txt_tmp " + str(dataFile) + "_ov
erlap_noHeader.txt > " + str(yFile) + "_overlap_noHeader.txt"

    #Join X and Y matrices with all combos of IDs
    join1 = "join -j 1 -t '\t' " + str(dataFileName) + "_noHeader_sort.txt " + str(yFileName) + "_noHeader_sort.txt > " + str(dataF
ileName) + "_" + str(yBaseName) + "_join.txt"

    #Cut mapped x matrix out into overlap file
    cut1 = "cut -f1-" + str(dataNumCols) + " " + str(dataFileName) + "_" + str(yBaseName) + "_join.txt > " + str(dataFileName) + "_
overlap_noHeader.txt"

    #Cut mapped y matrix out into overlap file
    cut2 = "cut -f1," + str(dataNumCols + 1) + "-" + str(dataNumCols + yNumCols - 1) + " " + str(dataFileName) + "_" + str(yBaseNam
e) + "_join.txt > " + str(yFileName) + "_overlap_noHeader.txt"

    #Create header files from original files
    header1 = "head -n 1 " + str(dataFile) + " > " + str(dataFileName) + "_header"
    header2 = "head -n 1 " + str(yFile) + " > " + str(yFileName) + "_header"

    #Add headers to new overlap files
    addHeader1 = "cat " + str(dataFileName) + "_header " + str(dataFileName) + "_overlap_noHeader.txt > " + str(dataFileName) + "_o
verlap.txt"
    addHeader2 = "cat " + str(yFileName) + "_header " + str(yFileName) + "_overlap_noHeader.txt > " + str(yFileName) + "_overlap.tx
t"

    #print(tail1+'\n')
    os.system(tail1)
    #print(tail2+'\n')
    os.system(tail2)
    #print(sort1+'\n')
    os.system(sort1)
    #print(sort2+'\n')
    os.system(sort2)
    #print(awk1+'\n')
    #os.system(awk1)
    #print(awk2+'\n')
    #os.system(awk2)
    #print(awk3+'\n')
    #os.system(awk3)
    #print(join1+'\n')
    os.system(join1)
    #print(cut1+'\n')
    os.system(cut1)
    #print(cut2+'\n')
    os.system(cut2)
    #print(header1+'\n')
    os.system(header1)
    #print(header2+'\n')
    os.system(header2)
    #print(addHeader1+'\n')
    os.system(addHeader1)
    #print(addHeader2+'\n')
    os.system(addHeader2)

	#Remove sample columns from x matrix and y vec files
    cut1 = "cut -f1 --complement " + str(dataFileName) + "_overlap.txt > " + str(dataFileName) + "_overlap_noSampleIDs.txt"
    cut2 = "cut -f1 --complement " + str(yFileName) + "_overlap.txt > " + str(yFileName) + "_overlap_noSampleIDs.txt"

    #print(cut1+'\n')
    os.system(cut1)
    #print(cut2+'\n')
    os.system(cut2)

dataFileNoIDs = str(dataFileName) + "_overlap_noSampleIDs.txt"
dataFile = str(dataFileName) + "_overlap.txt"
yFileNoIDs = str(yFileName) + "_overlap_noSampleIDs.txt"
yFile = str(yFileName) + "_overlap.txt"

time1 = time.process_time()
print("File manip time: ", time1-startTime,flush=True)



#Make sure that the number of nodes per iRF job evenly divides the total number of nodes for a script
if args.System == "Summit" and args.TotalNodes % args.NodesPer != 0:
	raise RuntimeError("NodesPer must evenly divide TotalNodes")

if args.System == "Brut" and args.TotalCores % args.CoresPer != 0:
        raise RuntimeError("CoresPer must evenly divide TotalCores")



#Overlap data header and index with sample IDs
dataArray = []
lineCount = 0
with open(dataFile, 'r') as df:
    for line in df.readlines():
        if lineCount == 0:
            topLineX = line.strip().split(delimiter)
        else:
            dataArray.append(line.strip().split(delimiter)[0])
        lineCount = lineCount + 1

dataArray = np.array(dataArray)
print("Data Array set",flush=True)
#y data header
topLine = []
with open(yFileNoIDs, 'r') as df:
    topLine = df.readline().strip().split(delimiter)

groups = []
if args.groupFile == "NAN" and args.Prediction:
    cut1 = "cut -f1 " + str(dataFile) + " > " + str(dataFileName) + "_sampleIDs.txt"
    print(cut1+'\n')
    os.system(cut1)
    groupFile = str(dataFileName) + "_sampleIDs.txt"
    #groups = []
    lineCount = 0
    with open(groupFile, 'r') as df:
        for line in df:
            if lineCount != 0:
                groups.append(line.strip())
            lineCount = lineCount + 1
elif args.groupFile != "NAN":
    groupFile = args.groupFile
    lineCount = 0
    with open(groupFile, 'r') as df:
        for line in df:
            if lineCount != 0:
                groups.append(line.strip())
            lineCount = lineCount + 1

time2 = time.process_time()
print("Make arrays time: ", time2-time1,flush=True)

#Make list of features
if args.ReRun:
	features = []
	with open(args.ReRunFile,'r') as rerunF:
		for line in rerunF:
			features.append(line.strip())

else:
    features = topLine

featureLists = []

currentDir = os.getcwd()


if not args.ReRun:
    #For each yvec, repeat kfold process
    currentDirList = [currentDir]*len(features)
    featureCountList = list(range(1,len(features)+1))
    foldSetsList = [args.foldSets]*len(features)
    dataArrayList = [dataArray]*len(features)
    groupsList = [groups]*len(features)
    sampleSizeList = [args.sampleSize]*len(features)
    kFoldList = [args.Kfold]*len(features)
    foldTypeList = [args.foldType]*len(features)
    dataFileNoIDsList = [dataFileNoIDs]*len(features)
    predictionList = [args.Prediction]*len(features)
    rerunList = [args.ReRun]*len(features)
    print("Entering feature runs",flush=True)
    with cc.ProcessPoolExecutor(max_workers=1) as executor:
        for feature,result in zip(features,executor.map(featureFoldRun,currentDirList,featureCountList,features,foldSetsList,dataAr
rayList,groupsList,sampleSizeList,kFoldList,foldTypeList,dataFileNoIDsList,predictionList,rerunList)):
            #currentDir, featureCount, feature, args.foldSets, dataArray, groups, args.sampleSize, args.Kfold,args.foldType, dataFi
leNoIDs, args.Prediction, args.ReRun
            print(feature,result)
        #time6A = time.process_time()
        #print("Feature time: ", time6A-time3)
time7 = time.process_time()
print("Full file manip time: ", time7-startTime)

if args.System == 'Summit':
    #Make a list of lists where each sublist contains the features that can run concurrently
    concurrentJobCount = args.TotalNodes / args.NodesPer

    try:
        currentDir = os.getcwd()
        newDir = currentDir + '/Submits'
        os.mkdir(newDir)
    except OSError:
        print("creation of the directory %s failed" % newDir)

    if args.Prediction:
        count = 1
        #Small list contains sublists of [feature, repeat, fold]
        smallList = []
        for featureCount in range(len(features)):
            for repeat in range(args.foldSets):
                for fold in range(args.Kfold):
                    #If concurrent set is full or it's the last feature, repeat, and set, add small list to full list
                    if count % concurrentJobCount == 0 or ((featureCount == len(features) -1) and (repeat == args.foldSets - 1) and
 (fold == args.Kfold -1)):
                        smallList.append([features[featureCount],repeat,fold])
                        featureLists.append(smallList)
                        #print("small list: ", len(smallList))
                        smallList = []
                    else:
                        smallList.append([features[featureCount],repeat,fold])
                    count = count + 1

        #for i in range(len(features)):
    	#	if count % concurrentJobCount != 0 and (i != len(features) - 1):
	    #		smallList.append(features[i])
	    #		print(features[i])
	    #	if count % concurrentJobCount == 0 or (i == len(features) - 1):
	    #		smallList.append(features[i])
	    #		featureLists.append(smallList)
	    #		smallList = []
	    #		print(features[i])
	    #	count = count + 1


	    #Calculate optimum number of sets to run in each script - based on Summit queue policies
        if args.TotalNodes < 46:
            totalSetsOneScript = 120 // (args.RunTime * 1.5)
            totalScripts = math.ceil(len(featureLists) / totalSetsOneScript)

        elif args.TotalNodes > 45 and args.TotalNodes < 92:
            totalSetsOneScript = 360 // (args.RunTime * 1.5)
            totalScripts = math.ceil(len(featureLists) / totalSetsOneScript)

        elif args.TotalNodes > 91 and args.TotalNodes < 922:
            totalSetsOneScript = 720 // (args.RunTime * 1.5)
            totalScripts = math.ceil(len(featureLists) / totalSetsOneScript)

        elif args.TotalNodes > 921:
            totalSetsOneScript = 1440 // (args.RunTime * 1.5)
            totalScripts = math.ceil(len(featureLists) / totalSetsOneScript)
        print(totalScripts)

	    #Make train scripts
        scriptLists = currentDir + '/Submits/submitAllTrain_' + str(args.RunName) + '.sh'
        featureListStart=0
        featureCount = 1
        for scriptCount in range(int(totalScripts)):
            outFile = currentDir + '/Submits/submit_train_' + str(args.RunName) +'_' + str(scriptCount) + '.sh'
            with open(outFile, 'w') as out:
                line1 = '#!/bin/bash -l\n'
                line2 = '#BSUB -P ' + str(args.Account) + '\n'
                if totalSetsOneScript > len(featureLists):
                    totalTime = math.ceil(len(featureLists) * args.RunTime * 1.5)
                else:
                    totalTime = totalSetsOneScript * args.RunTime * 1.5
                formatTime = '{:02d}:{:02d}'.format(*divmod(int(totalTime),60))
                line3 = '#BSUB -W ' + str(formatTime) + '\n'
                line4 = '#BSUB -nnodes ' + str(args.TotalNodes) + '\n'
                line5 = '#BSUB -J ' + str(args.RunName) + '_train_' + str(scriptCount) + '\n'
                line6 = '#BSUB -o ' + str(args.RunName) + '_train_' + str(scriptCount) + '.o%J\n'
                line7 = '#BSUB -e ' + str(args.RunName) + '_train_' + str(scriptCount) + '.e%J\n'
                line8 = '\n'
                line9 = 'export OMP_NUM_THREADS=160\n'
                line10 = '\n'
                line11 = 'cd ' + str(currentDir) + '\n'

                out.writelines([line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,line11])
                with open(scriptLists,'a') as scripts:
                    scripts.write('bsub ' + str(outFile) + '\n')

                featureSetCount = 0
                for featureSet in featureLists[featureListStart:]:
                    for feature in featureSet:
					    #Make results directory
                        treesPerNode = args.NumTrees / args.NodesPer

                        printLines = []

                        set1 = '/usr/bin/time -f "%e" jsrun -n ' + str(args.NodesPer) + ' -a 1 -c 40 -bpacked:40 /gpfs/alpine/syb10
5/proj-shared/Projects/iRF/IterativeRanger/cpp_version/build/ranger --file ' + currentDir + '/' + str(feature[0]) + '/foldRuns/fold
' + str(feature[1]) + "/Runs/Set" + str(feature[2]) + "/set" + str(feature[2]) + "_train_noSampleIDs.txt" + ' --yfile ' + currentDi
r + '/' + str(feature[0]) + '/foldRuns/fold' + str(feature[1]) + "/Runs/Set" + str(feature[2]) + "/set" + str(feature[2]) + "_Y_tra
in_noSampleIDs.txt"  + ' --treetype 3 --mtryType 1 --depvarname ' + str(feature[0]) + ' --ntree ' + str(int(treesPerNode)) + ' --ta
rgetpartitionsize ' + str(args.targetNodeSize) + ' --write --impmeasure 1 --nthreads 160 --useMPI 1 --numIterations ' + str(args.Nu
mIterations) + ' --outprefix ' + str(args.RunName) + '_' + str(feature[0]) + ' --printPathfile 0 '

                        printLines.append(set1)
                        if args.alwayssplitvars != "NAN":
                            set2 = "" + ' --alwayssplitvars ' + str(args.alwayssplitvars)
                            printLines.append(set2)

                        set3 = "" + ' --outputDirectory ' + currentDir + '/' + str(feature[0]) + '/foldRuns/fold' + str(feature[1])
 + "/Runs/Set" + str(feature[2]) + ' > ' + currentDir + '/' + str(feature[0]) + '/foldRuns/fold' + str(feature[1]) + "/Runs/Set" +
str(feature[2]) + '/' + str(args.RunName) + '_' + str(feature[0]) + '_train.o &'

                        printLines.append(set3)
                        out.write('\n')
                        out.writelines(printLines)
                        out.write('\n')

                        featureCount = featureCount + 1

                    out.write('wait \n')
                    featureSetCount = featureSetCount + 1
                    featureListStart = featureListStart + 1
                    if featureSetCount == totalSetsOneScript:
					    #featureListStart = featureSetCount
                        break

        time8 = time.process_time()
        #print("Make train script time: ", time8-time7)
        #Make Test scripts
        scriptLists = currentDir + '/Submits/submitAllTest_' + str(args.RunName) + '.sh'
        featureListStart = 0
        for scriptCount in range(int(totalScripts)):
            outFile = currentDir + '/Submits/submit_test_' + str(args.RunName) +'_' + str(scriptCount) + '.sh'
            with open(outFile, 'w') as out:
                line1 = '#!/bin/bash -l\n'
                line2 = '#BSUB -P ' + str(args.Account) + '\n'
                if totalSetsOneScript > len(featureLists):
                    totalTime = math.ceil(len(featureLists) * args.RunTime * 1.5)
                else:
                    totalTime = totalSetsOneScript * args.RunTime * 1.5
                formatTime = '{:02d}:{:02d}'.format(*divmod(int(totalTime),60))
                line3 = '#BSUB -W ' + str(formatTime) + '\n'
                line4 = '#BSUB -nnodes ' + str(args.TotalNodes) + '\n'
                line5 = '#BSUB -J ' + str(args.RunName) + '_test_' + str(scriptCount) + '\n'
                line6 = '#BSUB -o ' + str(args.RunName) + '_test_' + str(scriptCount) + '.o%J\n'
                line7 = '#BSUB -e ' + str(args.RunName) + '_test_' + str(scriptCount) + '.e%J\n'
                line8 = '\n'
                line9 = 'export OMP_NUM_THREADS=160\n'
                line10 = '\n'
                line11 = 'cd ' + str(currentDir)

                out.writelines([line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,line11])
                with open(scriptLists,'a') as scripts:
                    scripts.write('bsub ' + str(outFile) + '\n')

                featureCount = 0
                foldSetCount = 0
                featureSetCount = 0
                for featureSet in featureLists[featureListStart:]:
                    for feature in featureSet:
                        treesPerNode = args.NumTrees / args.NodesPer
                        line = '/usr/bin/time -f "%e" jsrun -n 1' + ' -a 1 -c 40 -bpacked:40 /gpfs/alpine/syb105/proj-shared/Projec
ts/iRF/IterativeRanger/cpp_version/build/ranger --file ' + currentDir + '/' + str(feature[0]) + '/foldRuns/fold' + str(feature[1])
+ "/Runs/Set" + str(feature[2]) + "/set" + str(feature[2]) + "_test_noSampleIDs.txt" + ' --yfile ' + currentDir + '/' + str(feature
[0]) + '/foldRuns/fold' + str(feature[1]) + "/Runs/Set" + str(feature[2]) + "/set" + str(feature[2]) + "_Y_test_noSampleIDs.txt"  +
 ' --predict ' + currentDir + '/' + str(feature[0]) +  '/foldRuns/fold' + str(feature[1]) + "/Runs/Set" + str(feature[2])+ '/' + st
r(args.RunName) + '_' + str(feature[0]) + '.forest' + ' --treetype 3 --depvarname ' + str(feature[0]) + ' --impmeasure 1 --nthreads
 160 --useMPI 0 --outprefix ' + str(args.RunName) + '_Set' + str(feature[2]) + '_test' + ' --outputDirectory ' + currentDir + '/' +
 str(feature[0]) + '/foldRuns/fold' + str(feature[1]) + "/Runs/Set" + str(feature[2]) + ' > ' + currentDir + '/' + str(feature[0])
+ '/foldRuns/fold' + str(feature[1]) +  "/Runs/Set" + str(feature[2]) + '/' + str(args.RunName) + '_Set' + str(feature[2]) + '_test
.o &'

                        out.write('\n')
                        out.write(line)
                        out.write('\n')

                        featureCount = featureCount + 1

                    out.write('wait \n')
                    featureSetCount = featureSetCount + 1
                    featureListStart = featureListStart + 1
                    if featureSetCount == totalSetsOneScript:
                        break
    time9 = time.process_time()
    #print("Make test file time: ", time9-time8)

    #Make full runs scripts
    featureLists = []
    count = 1
    #Small list contains feature names
    smallList = []
    for featureCount in range(len(features)):
        #If concurrent set is full or it's the last feature, add small list to full list
        if count % concurrentJobCount == 0 or (featureCount == len(features) -1):
            smallList.append(features[featureCount])
            featureLists.append(smallList)
            smallList = []
        else:
            smallList.append(features[featureCount])
        count = count + 1

    #Calculate optimum number of sets to run in each script - based on Summit queue policies
    if args.TotalNodes < 46:
        totalSetsOneScript = 120 // (args.RunTime * 1.5)
        totalScripts = math.ceil(len(featureLists) / totalSetsOneScript)

    elif args.TotalNodes > 45 and args.TotalNodes < 92:
        totalSetsOneScript = 360 // (args.RunTime * 1.5)
        totalScripts = math.ceil(len(featureLists) / totalSetsOneScript)

    elif args.TotalNodes > 91 and args.TotalNodes < 922:
        totalSetsOneScript = 720 // (args.RunTime * 1.5)
        totalScripts = math.ceil(len(featureLists) / totalSetsOneScript)

    elif args.TotalNodes > 921:
        totalSetsOneScript = 1440 // (args.RunTime * 1.5)
        totalScripts = math.ceil(len(featureLists) / totalSetsOneScript)
    print(totalScripts)

    scriptLists = currentDir + '/Submits/submitAllFull_' + str(args.RunName) + '.sh'
    featureListStart = 0
    for scriptCount in range(int(totalScripts)):
        outFile = currentDir + '/Submits/submit_full_' + str(args.RunName) +'_' + str(scriptCount) + '.sh'
        with open(outFile, 'w') as out:
            line1 = '#!/bin/bash -l \n'
            line2 = '#BSUB -P ' + str(args.Account) + '\n'
            if totalSetsOneScript > len(featureLists):
                totalTime = math.ceil(len(featureLists) * args.RunTime * 1.5)
            else:
                totalTime = totalSetsOneScript * args.RunTime * 1.5
            formatTime = '{:02d}:{:02d}'.format(*divmod(int(totalTime),60))
            line3 = '#BSUB -W ' + str(formatTime) + '\n'
            line4 = '#BSUB -nnodes ' + str(args.TotalNodes) + '\n'
            line5 = '#BSUB -J ' + str(args.RunName) + '_full_' + str(scriptCount) +'\n'
            line6 = '#BSUB -o ' + str(args.RunName) + '_full_' + str(scriptCount) + '.o%J\n'
            line7 = '#BSUB -e ' + str(args.RunName) + '_full_' + str(scriptCount) + '.e%J\n'
            line8 = '\n'
            line9 = 'export OMP_NUM_THREADS=160\n'
            line10 = '\n'
            line11 = 'cd ' + str(currentDir) +'\n'

            out.writelines([line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,line11])
            with open(scriptLists,'a') as scripts:
                scripts.write('bsub ' + str(outFile) + '\n')

            featureCount = 0
            foldSetCount = 0
            featureSetCount = 0
            for featureSet in featureLists[featureListStart:]:
                for feature in featureSet:
                    treesPerNode = args.NumTrees / args.NodesPer

                    printLines = []

                    set1 = '/usr/bin/time -f "%e" jsrun -n ' + str(args.NodesPer) + ' -a 1 -c 40 -bpacked:40 /gpfs/alpine/syb105/pr
oj-shared/Projects/iRF/IterativeRanger/cpp_version/build/ranger --file ' + str(dataFileNoIDs) + ' --yfile ' + currentDir + '/' + st
r(feature) + '/' + str(feature) + "_col.txt"  + ' --treetype 3 --mtryType 1 --depvarname ' + str(feature) + ' --ntree ' + str(int(t
reesPerNode)) + ' --targetpartitionsize ' + str(args.targetNodeSize) + ' --write --impmeasure 1 --nthreads 160 --useMPI 1 --numIter
ations ' + str(args.NumIterations) + ' --outprefix ' + str(args.RunName) + '_' + str(feature) + ' --printPathfile 0 '

                    printLines.append(set1)
                    if args.alwayssplitvars != "NAN":
                        set2 = "" + ' --alwayssplitvars ' + str(args.alwayssplitvars)
                        printLines.append(set2)

                    set3 = "" + ' --outputDirectory ' + currentDir + '/' + str(feature) + ' > ' + currentDir + '/' + str(feature) +
 '/' + str(args.RunName) + '_' + str(feature) + '.o &'

                    printLines.append(set3)
                    out.write('\n')
                    out.writelines(printLines)
                    out.write('\n')

                    featureCount = featureCount + 1

                out.write('wait \n')
                featureSetCount = featureSetCount + 1
                featureListStart = featureListStart + 1
                if featureSetCount == totalSetsOneScript:
                    #featureListStart = featureSetCount
                    break

    time10 = time.process_time()
    print("Make full file time: ", time10-time9)
    print("Total code time: ", time10-startTime)

if args.System == 'Brut':
	#Make a list of lists where each sublist contains the features taht can run concurrently
	concurrentJobCount = args.TotalCores / args.CoresPer
	count = 1
	smallList = []
	for i in range(len(features)):
		if count % concurrentJobCount != 0 and (i != len(features) -1):
			smallList.append(features[i])
		if count % concurrentJobCount == 0 or (i == len(features) -1):
			smallList.append(features[i])
			featureLists.append(smallList)
			smallList = []
		count = count + 1

	#Make directory to store submit scripts
	try:
		currentDir = os.getcwd()
		newDir = currentDir + '/Submits'
		os.mkdir(newDir)
	except OSError:
		print("creation of the directory %s failed" % newDir)

	#Make directory to store job results in
	try:
		newDir = currentDir + '/Runs'
		os.mkdir(newDir)
	except OSError:
		print("Creation of the directory %s failed" % newDir)

	totalSetsOneScript = 1440 // (args.RunTime * 1.5)
	print("totalSetsOneScript: \n",totalSetsOneScript)
	totalScripts = math.ceil(len(featureLists) / totalSetsOneScript)

	#Make Scripts
	scriptLists = currentDir + '/Submits/submitAll_' + str(args.RunName) + '.sh'
	featureListStart = 0
	for scriptCount in range(totalScripts):
		outFile = currentDir + '/Submits/submit' + str(scriptCount) + '.sh'
		with open(outFile,'w') as out:
			line1 = '#!/bin/bash \n'
			line2 = "#SBATCH --partition='brut_batch'\n"
			if len(featureLists) < totalSetsOneScript:
				totalTime = len(featureLists) * args.RunTime*1.5
			else:
				totalTime = totalSetsOneScript * args.RunTime*1.5
			#print("totalTime: \n", totalTime)
			formatTime = '{:02d}:{:02d}'.format(*divmod(int(totalTime),60))
			#print("formatTime: \n", formatTime)
			line3 = '#SBATCH -t ' + str(formatTime) + ':00\n'
			line4 = '#SBATCH -J ' + str(args.RunName) + '_' + str(scriptCount) + '\n'
			line5 = '#SBATCH --error="' + str(args.RunName) + '_' + str(scriptCount) + '.e%j"\n'
			line6 = '#SBATCH --output="' + str(args.RunName) + '_' + str(scriptCount) + '.o%j"\n'
			line7 = '#SBATCH --cpus-per-task=' + str(args.CoresPer) + '\n'
			line8 = '#SBATCH --ntasks ' + str(int(concurrentJobCount)) + '\n'
			line9 = '#SBATCH -N 1\n'
			line10 = '\n'
			line11 = 'module purge\n'
			line12 = 'module load gcc/5.3.0\n'
			line13 = 'module load cmake/3.15.2\n'
			line14 = 'module load openmpi/3.0.0\n'
			line15 = '\n'
			line16 = 'cd ' + str(currentDir) + '/Runs\n'

			out.writelines([line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,line11,line12,line13,line14,li
ne15,line16])

			with open(scriptLists,'a') as scripts:
				scripts.write('sbatch ' + str(outFile) + '\n')

			featureSetCount = 0
			for featureSet in featureLists[featureListStart:]:
				for feature in featureSet:
					#Make results directory
					try:
						newDir = currentDir + '/Runs/' + str(feature)
						os.mkdir(newDir)
					except OSError:
						print("Creation of the directory %s failed" % newDir)


					line = '/usr/bin/time -f "%e" mpirun -n 1 /lustre/or-hydra/cades-bsd/cji/iRF/IterativeRange
r/cpp_version/kcmbuild/ranger --file ' + str(args.DataFile) + ' --treetype 3 --depvarname ' + str(feature) + ' --ntree ' + str(args
.NumTrees) + "--targetpartitionsize " + str(args.targetNodeSize) + ' --impmeasure 1 --nthreads ' + str(args.CoresPer) + ' --useMPI
1 --numIterations ' + str(args.NumIterations) + ' --outprefix ' + str(args.RunName) + '_' + str(feature) + ' --printPathfile 0 --ou
tputDirectory ' + str(newDir) + ' > ' + str(newDir) + '/' + str(args.RunName) + '_' + str(feature) + '.o &'


					out.write('\n')
					out.write(line)
					out.write('\n')
				out.write('wait\n')
				featureSetCount = featureSetCount + 1
				featureListStart = featureListStart + 1
				if featureSetCount == totalSetsOneScript:
					#featureListStart = featureSetCount
					break
