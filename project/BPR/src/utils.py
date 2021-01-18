import pandas as pd
import scipy.sparse as sp
import numpy as np
import networkx as nx
from torch.utils import data
import torch


def load_data(dataset_str, random_split=True):
    data_dir = '../../yelp_data/'.format(dataset_str)
    df_train = pd.read_csv(data_dir + "train.csv")
    # df_test = pd.read_csv(data_dir + "test.csv")
    num_items = max(df_train.item.unique()) + 1
    num_users = max(df_train.user.unique()) + 1

    if random_split:
        df_test = df_train.sample(frac=0.2)
        test_set = df_test.groupby(['user'])['item'].apply(list).reset_index()
        train_interactions = df_train.drop(df_test.index)
    train_interactions = df_train[['user', 'item']].values
    test_set = df_test.groupby(['user'])['item'].apply(list).reset_index()
    
    social_data  = pd.read_csv(data_dir + "social.csv")
    social_dict = {}
    for  _,row in social_data.iterrows():
        user1,user2 = row['user1'],row['user2']
        if user1 not in social_dict:
           social_dict[user1] = [user2]
        else:
            social_dict[user1].append(user2)
    social = {key:torch.tensor(val) for key,val in social_dict.items()}
    return train_interactions, test_set, int(num_items), int(num_users), social


class Dataset(data.Dataset):
    def __init__(self, ratings):
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user, item = self.ratings[idx]
        return user, item
