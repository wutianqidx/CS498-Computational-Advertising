import pandas as pd
import scipy.sparse as sp
import numpy as np
import networkx as nx
from torch.utils import data
import torch


def load_data1(dataset_str, random_split=True):
    data_dir = '../../yelp_data/'.format(dataset_str)
    df_train = pd.read_csv(data_dir + "train.csv")
    # df_test = pd.read_csv(data_dir + "test.csv")
    num_items = max(df_train.item.unique()) + 1
    num_users = max(df_train.user.unique()) + 1

    if random_split:
        df_test = df_train.sample(frac=0.2)
        test_set = df_test.groupby(['user'])['item'].apply(list).reset_index()
        train_interactions = df_train.drop(df_test.index)[['user', 'item']].values
    else:
        train_interactions = df_train[['user', 'item']].values
        test_set = df_test.groupby(['user'])['item'].apply(list).reset_index()
    return train_interactions, test_set, int(num_items), int(num_users)

def load_data(dataset_str, random_split=True):
    data_dir = '../../yelp_data/'.format(dataset_str)
    df_train = pd.read_csv(data_dir + "train.csv")
    # df_test = pd.read_csv(data_dir + "test.csv")
    num_items = max(df_train.item.unique()) + 1
    num_users = max(df_train.user.unique()) + 1

    if random_split:
        df_test = df_train.sample(frac=0.2)
        test_set = df_test.groupby(['user'])['item'].apply(list).reset_index()
        train_interactions = df_train.drop(df_test.index)[['user', 'item']].values
        train_set = df_train.drop(df_test.index)[['user', 'item']].groupby(['user'])['item'].apply(list).reset_index()
    else:
        train_interactions = df_train[['user', 'item']].values
        test_set = df_test.groupby(['user'])['item'].apply(list).reset_index()
        train_set = df_train[['user', 'item']].groupby(['user'])['item'].apply(list).reset_index()

    return train_interactions, train_set, test_set, int(num_items), int(num_users)

class Dataset(data.Dataset):
    def __init__(self, ratings):
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user, item = self.ratings[idx]
        return user, item
