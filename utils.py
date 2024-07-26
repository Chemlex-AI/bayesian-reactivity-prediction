import numpy as np

from scipy.stats import entropy
from drfp import DrfpEncoder


def cal_DRFP(rxn_smis, nbits=2048):
    """

    Computes the DRFP of a list of reaction smiles
    The reaction smiles should be formulated in REACTANTS.REAGENTS>>PRODUCTS
    Parameters:
        rxn_smis - the list of reaction smiles
    Returns:
        fps - the drfp with nbits length 
    """
    fps = DrfpEncoder.encode(rxn_smis, n_folded_length=nbits)
    return fps


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def disentangle_uncertainty(pred_list):
    """ Disentangle different types of uncertainties from a list of predictions.

    Parameters:
        pred_list: A list of prediction probabilities, where each element from different sampled models.

    Returns:
        mean: The mean of the prediction probabilities.
        predictive_uncertainty: The predictive uncertainty.
        aleatoric_uncertainty: The aleatoric uncertainty.
        epistemic_uncertainty: The epistemic uncertainty.
    """

    mean = np.mean(pred_list, 1)
    mean_probs = np.stack([mean, 1 - mean])

    # Calculate the predictive uncertainty
    predictive_uncertainty = entropy(mean_probs, axis=0)

    # Calculate the mean aleatoric uncertainty across all predictions
    aleatoric_uncertainty = []
    for pred in pred_list:
        ent_per_inf = []
        for p in pred:
            ent_per_inf.append(entropy([p, 1-p]))
        aleatoric_uncertainty.append(np.mean(ent_per_inf))
    aleatoric_uncertainty = np.array(aleatoric_uncertainty)

    # Calculate the epistemic uncertainty
    epistemic_uncertainty = predictive_uncertainty - aleatoric_uncertainty

    return mean, predictive_uncertainty, aleatoric_uncertainty, epistemic_uncertainty.clip(0, 1)
