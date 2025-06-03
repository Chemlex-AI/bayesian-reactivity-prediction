# Bayesian-reactivity-prediction

## Overview
Code & Data release of paper:

[**Towards Global Feasibility Prediction and Robustness Estimation of Organic Chemical Reactions with High Throughput Experimentation Data and Bayesian Deep Learning**](https://chemrxiv.org/engage/chemrxiv/article-details/66a8e186c9c6a5c07a7f6966)

## Introduction
This repo provides reactivity prediction and uncertainty estimation for wetlab data. The project implements a binary classification model to predict chemical reaction feasibility, featuring five uncertainty estimation methods:

- Bayesian Neural Network (SVI and NUTS)
- MCDropout
- Ensemble
- Deep Kernel Learning Gaussian Process

## Installation & Setup

### Requirements
```bash
# create a custom conda environment
conda create --name myenv python=3.9

# activate the conda environment
conda activate myenv

# install the required packages
python -m pip install -r requirements.txt
```

### Project Structure
```
.
├── data/                           # Data directory
│   ├── wetlab/                    # Wetlab experimental data
│   │   ├── Chemlex_Acidamine_Wetlab_Data.xlsx
│   │   ├── wetlab_drfp.npy       
│   │   ├── wetlab_GNN.npy        
│   │   └── wetlab_rxnfp.npy      
│   └── public/                    # Public dataset
│       ├── suzuki_data.xlsx      
│       ├── suzuki_drfp.npy       
│       ├── suzuki_GNN.npy        
│       └── suzuki_rxnfp.npy      
├── jax_model.py                   # Bayesian Neural Network (NUTS & SVI)
├── torch_model.py                 # MCDropout, Ensemble, DKLGP
├── utils.py                       # Utility functions
├── train.py                       # Main training script
├── inference.py                   # Inference script
├── requirements.txt               # Python dependencies
└── README.md
```

## Data

### Dataset Information
The complete wetlab data are available [here](https://doi.org/10.5281/zenodo.15401035).

We also provide public Suzuki HTE data, curated from datasets published by [D. T. Ahneman *et al.*](https://www.science.org/doi/10.1126/science.aar5169) and [A. B. Santanilla *et al.*](https://www.science.org/doi/10.1126/science.1259203), and utilized in a publication by [P. Schwaller](https://rxn4chemistry.github.io/rxn_yields/).

You can download all used features [here](https://drive.google.com/drive/folders/1yAW-vPn8cIvr2b8iHesoaLLWM3BKA3kl?usp=drive_link) and put them in `data/`.


### Available Split Types
| Dataset Split | Description |
|--------------|-------------|
| Random_Split | Randomly split the dataset into training/test set (70%/30%).                                                                        |
| Stratified_Split_One_Unseen | Split the dataset into training/test set (70%/30%) with only one test reactant available in the training set. |
|Stratified_Split_Both_Unseen| Split the dataset into training/test set (70%/30%) with no overlap between both reactants (no test reactants in the training set).|



## Usage

### Model Types and Configurations
| Model Type | Description | Configuration | Key Features |
|------------|-------------|---------------------|--------------|
| BNN_NUTS | BNN with No U-Turn Sampler | - | - Full Bayesian inference<br>- MCMC sampling<br>- Best uncertainty estimates |
| BNN_SVI | BNN with Stochastic Variational Inference | lr: 0.1| - Variational inference<br>- Faster than NUTS<br>- Good balance of speed/accuracy |
| MCDropout | Monte Carlo Dropout | lr: 3e-3<br>dropout_rate: 0.3 | - Simple implementation<br>- Dropout at test time<br>- Approximate Bayesian inference |
| Ensemble | Deep Ensemble | lr: 3e-4| - Multiple models trained independently<br>- Robust predictions |
| DKLGP | GP with Deep Kernel Learning | lr: 3e-3<br>base_kernel: Matern52 | - Variational Sparse GP<br>- Non-parametric Bayesian model<br>- Deep Kernel Learning |


### Training
```bash
# BNN_NUTS
python train.py --model_type BNN_NUTS --split_type <SPLIT_TYPE> --seed 666

# BNN_SVI
python train.py --model_type BNN_SVI --lr 0.1 --split_type <SPLIT_TYPE> --seed 666

# MCDropout
python train.py --model_type MCDropout --lr 3e-3 --split_type <SPLIT_TYPE> --seed 666

# Ensemble
python train.py --model_type Ensemble --lr 3e-4 --split_type <SPLIT_TYPE> --seed 666

# Deep Gaussian Process
python train.py --model_type DKLGP --lr 3e-3 --split_type <SPLIT_TYPE> --seed 666
```

### Output Format
The training process outputs the model's performance on the test set, including accuracy, F1-score , and AUC-ROC.

Additionally, training generates `{split_type}_{model_type}_predictions_uncertainty.csv` with:

| Column Name | Description |
|------------|-------------|
| Conversion | Experimental conversion value |
| label | Ground truth (0: infeasible, 1: feasible) |
| pred | Predicted feasibility |
| aleatoric_unc | Data noise uncertainty |
| epistemic_unc | Model knowledge uncertainty |
| predictive_unc | Total uncertainty |
| logits | Raw model outputs |



### Inference
You can download the trained models [here](https://drive.google.com/drive/folders/1yAW-vPn8cIvr2b8iHesoaLLWM3BKA3kl?usp=drive_link).
```bash
python inference.py --rxn <RXN_SMILES>

# python inference.py --rxn 'O=C(O)COCCOCCOCCNC(=O)OCC1c2ccccc2-c2ccccc21.C[C@H](N)c1cccc([N+](=O)[O-])c1.CCN(C(C)C)C(C)C.F[P-](F)(F)(F)(F)F.c1ccc2c(c1)nnn2O[P+](N1CCCC1)(N1CCCC1)N1CCCC1>>C[C@H](NC(=O)COCCOCCOCCNC(=O)OCC1c2ccccc2-c2ccccc21)c1cccc([N+](=O)[O-])c1' 'CC(C(=O)O)c1cccc(C(=O)c2ccccc2)c1.COC(=O)c1nc(N)sc1Br.CCN(C(C)C)C(C)C.CN(C)C(On1nnc2cccnc21)=[N+](C)C.F[P-](F)(F)(F)(F)F>>COC(=O)c1nc(NC(=O)C(C)c2cccc(C(=O)c3ccccc3)c2)sc1Br'

# Returns: [1,0] (1: feasible, 0: infeasible)
```

## Contributing

### Getting Started
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes
4. Submit pull request

### Issues
- Use GitHub issue tracker
- Provide clear descriptions
- Include reproduction steps
- Add system information
