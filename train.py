import os
import argparse
import random

import torch
import pyro
import numpy as np
import pandas as pd

from torch_model import MCDropout, Ensemble, DKLGP
from jax_model import BNN_NUTS, BNN_SVI
from utils import data_process, Metrics


def seed_all(seed=666):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pyro.set_rng_seed(seed)


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Hardware settings
    parser.add_argument("--gpu_devices", type=str, default="0,1,2,3",
                        help="GPU device indices to use")

    # Data settings
    parser.add_argument("--data_path", type=str,
                        default="./data/wetlab/Chemlex_Acidamine_Wetlab_Data.xlsx",
                        help="Path to input data file")
    parser.add_argument("--fps_path", type=str,
                        default="./data/wetlab/Wetlab_drfp.npy",
                        help="Path to fingerprint data file")

    # Model settings
    parser.add_argument("--model_type", type=str, default="MCDropout",
                        choices=["BNN_SVI", "MCDropout", "Ensemble",
                                 "DKLGP", "BNN_NUTS"],
                        help="Type of model to train")
    parser.add_argument("--hidden_dim", type=int, default=16,
                        help="Hidden layer dimension")
    parser.add_argument("--dropout_rate", type=float, default=0.3,
                        help="Dropout rate for MCDropout")

    # Training settings
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-3,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=666,
                        help="Random seed")

    # Data split settings
    global split_choices
    split_choices = [f"k_fold_{i}" for i in range(10)]
    split_choices.extend(["Random_Split", "Stratified_Split_One_Unseen",
                         "Stratified_Split_Both_Unseen"])
    parser.add_argument("--split_type", type=str, default="Random_Split",
                        choices=split_choices,
                        help="Type of data split to use")

    # Output settings
    parser.add_argument("--save_test", type=bool, default=True,
                        help="Whether to save test predictions")

    # Additional options
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Additional configuration options")

    return parser.parse_args()


def main(args):
    """Main training function."""
    # Print arguments
    print("="*100)
    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")
    print("="*100)

    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    seed_all(seed=args.seed)

    # Load data
    df = pd.read_excel(args.data_path)
    fps = np.load(args.fps_path)
    labels = (df["Conversion"] >= 20).astype(int).tolist()

    # Process data
    train_x, train_y, test_x, test_y, train_loader = data_process(
        df, fps, labels, args.split_type, args.model_type)
    print(f"Training size: {train_y.shape[0]}, Test size: {test_y.shape[0]}")

    # Model configuration
    model_params = {
        'input_dim': 2048,
        'hidden_dim': args.hidden_dim,
        'output_dim': 1,
        'lr': args.lr,
        'dropout_rate': args.dropout_rate,
        'epochs': args.epochs,
    }

    # Initialize model
    if args.model_type == "BNN_SVI":
        net = BNN_SVI(model_params['input_dim'], model_params['hidden_dim'],
                      model_params['output_dim'], lr=model_params['lr'])
    elif args.model_type == "BNN_NUTS":
        net = BNN_NUTS(model_params['input_dim'], model_params['hidden_dim'],
                       model_params['output_dim'])
    elif args.model_type == "MCDropout":
        net = MCDropout(model_params['input_dim'], model_params['hidden_dim'],
                        model_params['output_dim'],
                        model_params['dropout_rate'])
    elif args.model_type == "Ensemble":
        net = Ensemble(model_params['input_dim'], model_params['hidden_dim'],
                       model_params['output_dim'], 100)
    elif args.model_type == "DKLGP":
        net = DKLGP(model_params['input_dim'], model_params['hidden_dim'],
                    model_params['output_dim'],
                    train_x.shape[0], train_loader)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # Train model
    if "BNN" in args.model_type:
        net.train(train_x, train_y)
    else:
        net.train(args.epochs, train_loader, model_params['lr'])

    # Evaluate model
    test_preds, all_unc, data_unc, model_unc = net.evaluate(test_x, 100)
    metrics = Metrics(test_y, test_preds)
    acc, f1_macro, auc_roc = metrics.metric_for_binary_classification()

    # Print results
    print(f"Test accuracy: {acc * 100:.2f}%")
    print(f"Test F1_macro: {f1_macro:.2f}")
    print(f"Test AUC-ROC: {auc_roc:.2f}")

    # Save results if requested
    if args.save_test:
        df_test = df[df[args.split_type] == "test"].reset_index(drop=True)
        df_test["label"] = (df_test["Conversion"] >= 20).astype(int)
        df_test["pred"] = (test_preds > 0.5).astype(int)
        df_test["aleatoric_unc"] = data_unc
        df_test["epistemic_unc"] = model_unc
        df_test["predictive_unc"] = all_unc
        df_test["logits"] = test_preds

        df_test = df_test.drop(
            columns=[col for col in split_choices if col in df_test.columns])
        output_path = f"./{args.split_type}_{args.model_type}_predictions_uncertainty.csv"
        df_test.to_csv(output_path, index=False)

    print("="*100)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
