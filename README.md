# Popularity-Bias-in-Dynamic-Recommendation
Code for the KDD21 paper -- Popularity Bias in Dynamic Recommendation

## Data
We put the pre-processed MovieLens 1M dataset in the 'Data' folder, where 1000 users are randomly selected for the simulation experiments. We run an MF model with cross entropy loss to complete the user-item relevance matrix to get the ground truth data for the simulation experiments. In each 'Experiment' folder, the outputs of the simulation experiments are stored.

## Requirements
python 3, tensorflow 1.14.0, numpy, pandas

## Excution
To run the similation experiment with different settings, use the command 'python3 Experiment_XXX.py'. For example, to run the basic simulation experiment with vanilla MF model, position bias, closed feedback loop, better negative sampling, use the command 'python3 Experiment_basic.py'.

**Experiment_basic**: with vanilla MF, position bias, closed feedback loop, better negative sampling.  
**Experiment_withoutPB**: with vanilla MF, position bias counteracted, closed feedback loop, better negative sampling.  
**Experiment_withoutCFL**: with vanilla MF,  position bias counteracted, without closed feedback loop, better negative sampling.  
**Experiment_vanillaNS**: with vanilla MF, position bias counteracted, closed feedback loop, vanilla negative sampling.  
**Experiment_Scale**: based on Experiment_withoutPB, apply static Scale method to debias.  
**Experiment_DScale**: based on Experiment_withoutPB, apply the proposed dynamic method DScale to debias.   
**Experiment_FPC**: based on Experiment_withoutPB, apply the proposed False Positive Correction FPC to debias.   
**Experiment_FPC_DScale**: based on Experiment_withoutPB, apply the proposed FPC + DScale to debias.   


After running the simulations, to plot the results, go to the 'Experiment' folders in './Data/ml1m', and run the 'analysis.ipynb' to see the ploting results. For example, to see the results of Experiment_basic, go to './Data/ml1m/Experiment_basic', and run 'analysis.ipynb'.



