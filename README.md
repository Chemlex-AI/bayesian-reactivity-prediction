# Bayesian-reactivity-prediction
Code & Data release of paper:

[**Towards Global Feasibility Prediction and Robustness Estimation of Organic Chemical Reactions with High Throughput Experimentation Data and Bayesian Deep Learning**](https://chemrxiv.org/engage/chemrxiv/article-details/66a8e186c9c6a5c07a7f6966)

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
The wetlab data are currently available upon request and will be released in `data/Chemlex_Acidamine_Wetlab_Data.xlsx` following the publication of this work (For now, only conversion and split type data are available in this repository for running demos).


#### Available Split Types
| Dataset Split                                                                           | Description                                                                                                                                   |
|-------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Random_Split | Randomly split the dataset into training/test set (70%/30%).                                                                        |
| Stratified_Split_One_Unseen | Split the dataset into training/test set (70%/30%) with only one test reactant available in the training set. |
|Stratified_Split_Both_Unseen| Split the dataset into training/test set (70%/30%) with no overlap between both reactants (no test reactants in the training set).|

### Train

#### Run Demos

```bash
# BNN_NUTS
python train.py --model_type BNN_NUTS --split_type <SPLIT_TYPE>

# BNN_SVI
python train.py --model_type BNN_SVI --lr 0.1 --split_type <SPLIT_TYPE>

# MCDropout
python train.py --model_type MCDropout --lr 3e-3 --split_type <SPLIT_TYPE>

# Ensemble
python train.py --model_type Ensemble --lr 3e-4 --split_type <SPLIT_TYPE>

# Deep Gaussian Process
python train.py --model_type Deep_GP --lr 3e-3 --split_type <SPLIT_TYPE>
```
A dataframe will be generated under this path, including predictions and uncertainties (predictive, aleatoric and epistemic) for all samples.


<!-- ## Citation

If you find our work useful in your research, please consider citing:

```

``` -->


