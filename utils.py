import torch
import numpy as np
import sklearn.metrics as metrics

from scipy.stats import entropy
from drfp import DrfpEncoder

from torch.utils.data import DataLoader, TensorDataset


class Metrics(object):
    def __init__(self, groundtruth: np.array, logits: np.array) -> None:
        self.gt = groundtruth
        self.logits = logits
        self.preds = logits > 0.5

    def metric_for_binary_classification(self):
        acc = metrics.accuracy_score(self.gt, self.preds)
        f1_macro = metrics.f1_score(self.gt, self.preds, average="macro")
        auc_roc = metrics.roc_auc_score(self.gt, self.logits)

        return acc, f1_macro, auc_roc


def data_process(df, fps, labels, set_col, model_type):
    input_data = fps.astype(np.float32)

    if model_type == "BNN_SVI":
        input_data = data_standardization(input_data)

    labels = np.array(labels).astype(np.float32)
    train_inds = df[df[set_col] == "train"].index.values
    test_inds = df[df[set_col] == "test"].index.values
    if "BNN" not in model_type:
        train_x = torch.tensor(input_data[train_inds, ...])
        train_y = torch.tensor(labels[train_inds])
        test_x = torch.tensor(input_data[test_inds, ...])
        test_y = torch.tensor(labels[test_inds])

        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

        return train_x, train_y, test_x, test_y, train_loader

    else:
        train_x = input_data[train_inds, ...]
        train_y = labels[train_inds]
        test_x = input_data[test_inds, ...]
        test_y = labels[test_inds]

        return train_x, train_y, test_x, test_y, None


def data_standardization(input_data):
    input_data = input_data[:, np.where(input_data.std(axis=0) != 0)[0]]

    mean = np.mean(input_data, axis=0)
    std = np.std(input_data, axis=0)

    standardized_data = (input_data - mean) / std

    return standardized_data


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

    mean = np.round(np.mean(pred_list, 1), 3)
    mean_probs = np.stack([mean, 1 - mean])

    # Calculate the predictive uncertainty
    predictive_uncertainty = entropy(mean_probs, axis=0)
    predictive_uncertainty = np.round(predictive_uncertainty, 3)
    # Calculate the mean aleatoric uncertainty across all predictions
    aleatoric_uncertainty = []
    for pred in pred_list:
        ent_per_inf = []
        for p in pred:
            ent_per_inf.append(entropy([p, 1-p]))
        aleatoric_uncertainty.append(np.mean(ent_per_inf))
    aleatoric_uncertainty = np.round(np.array(aleatoric_uncertainty), 3)

    # Calculate the epistemic uncertainty
    epistemic_uncertainty = np.round(
        predictive_uncertainty - aleatoric_uncertainty, 3)

    return mean, predictive_uncertainty, aleatoric_uncertainty, epistemic_uncertainty
