import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


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
