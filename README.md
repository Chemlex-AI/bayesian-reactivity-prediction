# Bayesian-reactivity-prediction
Code & Data(soon) release of paper:

[**Towards Global Feasibility Prediction and Robustness Estimation of Organic Chemical Reactions with High Throughput Experimentation Data and Bayesian Deep Learning**]()

## Introduction

This repo provides reactivity prediction and uncertainty estimation for wetlab data. 

Specifically, a binary classification model is trained to predict whether a chemical reaction is feasible or not. Five uncertainty estimation methods are provided for now:

- Bayesian Neural Network (SVI and NUTS)
- MCDropout
- Ensemble
- Deep Gaussian Process

## Quick Start
### Requirements
Create your virtual environment and install the required packages
```bash
# in your virtual environment
pip install -r requirements.txt
```
### Data
The wetlab data are avaliable upon request now and will be release in `data/Chemlex_Acidamine_Wetlab_Data.xlsx` after publication of this work.

| Dataset Split                                                                           | Description                                                                                                                                   |
|-------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Random_Split | Randomly split the dataset into training/test set (70%/30%).                                                                        |
| Stratified_Split_One_Unseen | Split the dataset into training/test set (70%/30%) with only one reactant available in the training set. |
|Stratified_Split_Both_Unseen|Split the dataset into training/test set (70%/30%) with no overlap between both reactants (no test reactants in training set).|

### Train
```bash
# BNN_NUTS
python train.py --model_type BNN_NUTS  --split_type 
# BNN_SVI
python train.py --model_type BNN_SVI --lr 0.1
# MCDropout
python train.py --model_type MCDropout --lr 3e-3
# Ensemble
python train.py --model_type Ensemble --lr 3e-4
# Deep Gaussian Process
python train.py --model_type Deep_GP --lr 3e-3
```
A dataframe will be generated under this path, including predictions and uncertainty (predictive, aleatoric and epistemic) for all samples.


<!-- ## Citation

If you find our work useful in your research, please consider citing:

```

``` -->


