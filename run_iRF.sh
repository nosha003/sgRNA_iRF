# Builder script: /gpfs/alpine/syb105/proj-shared/Projects/iRF/iRF_LOOP_SetUp_CrossLayer.py
# [python iRF_LOOP_SetUp_CrossLayer.py --DataFile --YFile --System XX --NodesPer 1 --TotalNodes 10 --RunTime 2 --Account XX --NumTrees 1000 --NumIterations 5 --RunName iRF.XX --bypass --Prediction]

module load python/3.7-anaconda3

mkdir <data_path>
cd <data_path>
# xmat = features matrix
# yvec = vector of real values for training prediction model
python iRF_LOOP_SetUp_CrossLayer.py --System XX --NodesPer 1 --TotalNodes 50 --RunTime 90 --Account XX --NumTrees 1000 --NumIterations 5 --RunName run_name --bypass --targetNodeSize 50 --Prediction xmat.txt yvec.txt

# run Submit scripts (full, train, test)

# post-processing 
# YNames is a text file with the yvec column name
python iRF_postProcessing.py --Iterations 5 --Prediction --PredAccuracy MAE,MAEA,MSE,R2 --varTot 95 YNames.txt run_name
