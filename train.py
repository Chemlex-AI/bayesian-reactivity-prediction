import os
import argparse
import random
import torch
import pyro


import numpy as np
import pandas as pd


from torch_model import MCDropout, Ensemble, Deep_GP
from jax_model import BNN_NUTS, BNN_SVI
from data_utils import data_process
from metric import Metrics


def seed_all(seed=666):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pyro.set_rng_seed(seed)


def _parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--gpu_devices", type=str, default="0")
    parser.add_argument("--data_path", type=str,
                        default="./data/Chemlex_Acidamine_Wetlab_Data.xlsx")
    parser.add_argument("--fps_path", type=str,
                        default="./data/Wetlab_drfp.npy", required=False)
    parser.add_argument("--epochs", type=int, default=30, required=False)
    parser.add_argument("--hidden_dim", type=int, default=16, required=False)
    parser.add_argument("--dropout_rate", type=float,
                        default=0.3, required=False)
    parser.add_argument("--lr", type=float, default=0.1, required=False)
    parser.add_argument("--seed", type=int, default=666, required=False)
    parser.add_argument("--save_test", type=bool, default=True, required=False)
    parser.add_argument("--split_type", type=str, default="Random_Split",
                        choices=["Random_Split", "Stratified_Split_One_Unseen", "Stratified_Split_Both_Unseen"], required=False)
    parser.add_argument("--model_type", type=str, default="BNN_NUTS",
                        choices=["BNN_SVI", "MCDropout", "Ensemble", "Deep_GP", "BNN_NUTS"], help="model type")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    return parser.parse_args()


def main(args):
    print("="*100)
    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")
    print("="*100)

    seed_all(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    df = pd.read_excel(args.data_path)
    fps = np.load(args.fps_path)
    labels = (df["Conversion"] >= 20).astype(int).tolist()

    train_x, train_y, test_x, test_y, train_loader = data_process(
        df, fps, labels, args.split_type, args.model_type)
    print(
        f"Train labels shape: {train_y.shape}, Test labels shape: {test_y.shape}")

    # Initialize model
    model_params = {
        'input_dim': 2048,
        'hidden_dim': args.hidden_dim,
        'output_dim': 1,
        'lr': args.lr,
        'dropout_rate': args.dropout_rate,
        'epochs': args.epochs,
    }

    if args.model_type == "BNN_SVI":
        net = BNN_SVI(model_params['input_dim'], model_params['hidden_dim'],
                      model_params['output_dim'], lr=model_params['lr'])
    elif args.model_type == "BNN_NUTS":
        net = BNN_NUTS(
            model_params['input_dim'], model_params['hidden_dim'], model_params['output_dim'])
    elif args.model_type == "MCDropout":
        net = MCDropout(model_params['input_dim'], model_params['hidden_dim'],
                        model_params['output_dim'], model_params['lr'], model_params['dropout_rate'])
    elif args.model_type == "Ensemble":
        net = Ensemble(model_params['input_dim'], model_params['hidden_dim'],
                       model_params['output_dim'], model_params['lr'], 100)
    elif args.model_type == "Deep_GP":
        net = Deep_GP(model_params['input_dim'], model_params['hidden_dim'],
                      model_params['output_dim'], model_params['lr'], train_x.shape[0], train_loader)
    else:
        raise ValueError("Unsupported model type")

    if "BNN" in args.model_type:
        net.train(train_x, train_y)
    else:
        net.train(args.epochs, train_loader)

    test_preds, all_unc, data_unc, model_unc = net.evaluate(test_x, 100)

    m = Metrics(test_y, test_preds)
    acc, f1_macro, auc_roc = m.metric_for_binary_classification()
    print(f"Test accuracy: {acc * 100:.2f}%")
    print(f"Test F1_macro: {f1_macro:.2f}")
    print(f"Test AUC-ROC: {auc_roc:.2f}")

    if args.save_test:
        df_test = df[df[args.split_type] == "test"].reset_index(drop=True)
        df_test["aleatoric_unc"] = data_unc
        df_test["epistemic_unc"] = model_unc
        df_test["predictive_unc"] = all_unc
        df_test["pred"] = test_preds
        df_test.to_csv(
            f"./{args.split_type}_{args.model_type}_predictions_uncertainty.csv", index=False)

    print("="*100)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
