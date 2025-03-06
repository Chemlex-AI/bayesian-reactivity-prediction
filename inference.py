import argparse

import numpy as np
import jax.numpy as jnp
import jax.random as random

from numpyro.infer import Predictive

from jax_model import BNN_NUTS
from utils import cal_DRFP, disentangle_uncertainty


def init_model(model_path="./models/BNN_NUTS.npz"):
    model = BNN_NUTS(2048, 16, 1)
    with np.load(model_path, mmap_mode='r') as data:
        model_samples = {key: jnp.array(data[key]) for key in data.keys()}
    predictive_probs = Predictive(
        model.jax_model, posterior_samples=model_samples, return_sites=["prob"])
    return predictive_probs


def inf(rxn_lst, predictive_probs, seed=666, verbose=False):
    fps = jnp.array(cal_DRFP(rxn_lst))
    _, rng_key_predict = random.split(random.PRNGKey(seed))
    probs_posterior = np.array(predictive_probs(
        rng_key_predict, fps, None, 16)["prob"])
    predictions = probs_posterior.T
    mean, predictive_uncertainty, aleatoric_uncertainty, epistemic_uncertainty = disentangle_uncertainty(
        predictions)
    if verbose:
        print(f"mean: {mean}, \npredictive_uncertainty: {predictive_uncertainty}, \naleatoric_uncertainty: {aleatoric_uncertainty}, \nepistemic_uncertainty: {epistemic_uncertainty}")
    flag = (mean > 0.5).astype(int)
    return flag


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference for reaction feasibility")
    parser.add_argument('--rxn', nargs='+', required=True,
                        help='List of reaction SMILES strings')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed uncertainty information')
    args = parser.parse_args()

    predictive_probs = init_model()
    flag = inf(args.rxn, predictive_probs, seed=666, verbose=args.verbose)
    print("Prediction: ", flag)
