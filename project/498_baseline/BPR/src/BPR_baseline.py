import torch.nn as nn
import torch.nn.functional as F
from utils import *

class BPR(nn.Module):
    def __init__(self, num_items, num_users, embed_dim, lambda_u=0.01, lambda_i=0.01,
    method='BPR', use_gpu=0,social=0):
        super(BPR, self).__init__()
        self.num_items = num_items
        self.num_users = num_users
        print ("# users", self.num_users, "# items", self.num_items)
        self.embed_dim = embed_dim
        self.method = method
        self.lambda_u = lambda_u
        self.lambda_i = lambda_i
        self.use_gpu = use_gpu
        self.social = social

        self.item_embeddings = nn.Embedding(self.num_items, self.embed_dim)
        nn.init.xavier_uniform_(self.item_embeddings.weight)

        if self.method == 'BPR':
            self.user_embeddings = nn.Embedding(self.num_users, self.embed_dim)
            nn.init.xavier_uniform_(self.user_embeddings.weight)

    def forward(self, users, items):
        assert users.shape[0] == items.shape[0]
        batch_size = users.shape[0]

        if self.method == 'BPR':
            batch_user_embeddings = self.user_embeddings(users)  # [B, D]
            # batch_user_embeddings = self.user_embeddings[users]
        # user_embeddings = [B, D]
        batch_item_embeddings = self.item_embeddings(items)  # [B, D]
        self.batch_user_embeddings = batch_user_embeddings
        # [B, D] x [B, D, 1] -> [B, 1] -> [B]
        positive_predictions = torch.bmm(batch_user_embeddings.view(batch_size, 1, self.embed_dim),
                                         batch_item_embeddings.view(batch_size, self.embed_dim,
                                                                    1)).squeeze()
        return positive_predictions

    def sample_negs(self, items, num_negs):
        batch_size = items.shape[0]
        negative_items = torch.randint(0, self.num_items, (batch_size, num_negs))  # [B, 10]
        return negative_items


    def loss(self, positive_predictions , negative_predictions):
        # BPR loss.
        bpr_loss =  - torch.log(torch.sigmoid(positive_predictions -
                                negative_predictions)).mean()
        # regularization loss.
        if self.method == 'BPR':
            bpr_loss += self.lambda_u * torch.norm(self.user_embeddings.weight)**2
            bpr_loss += self.lambda_i * torch.norm(self.item_embeddings.weight)**2

        return bpr_loss
