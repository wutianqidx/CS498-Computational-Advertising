import models
import argparse
from tqdm import tqdm
import torch.optim as optim
import networkx as nx
import torch
from torch.utils import data
import sys, os

import utils
from eval_metrics import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def evaluate(test_set, train_set, model, mode = 'test',use_gpu = 0):
    rec_list = {}
    actual_list = {}
    user_scores = {}
    # Evaluation is done for each user one at a time.
    # TODO: Optimize to evaluate batch-wise.
    for u, item_set in test_set.values:
        input_users = torch.LongTensor([u])
        if use_gpu:
            input_users = input_users.cuda()

        user_embed = model.user_embeddings(input_users)
        item_range = torch.arange(0, model.num_items)
        if use_gpu:
            item_range = item_range.cuda()
        scores = torch.mm(user_embed, model.item_embeddings(item_range).t()).squeeze()
        user_scores[u] = scores.detach().cpu().numpy()
        #_, rec_list_u = scores.topk(100, dim=0)
        #rec_list[u] = rec_list_u.cpu().numpy().tolist()
        #actual_list[u] = dict(zip(item_set, item_set))
        _, rec_list_u = scores.topk(2000, dim=0)
        train_list = set(train_set[train_set['user'] == u]['item'].values[0])
        rec_list_u = rec_list_u.cpu().numpy().tolist()
        rec_list_u = [i for i in rec_list_u if i not in train_list]
        rec_list[u] = rec_list_u[:100]
        actual_list[u] = dict(zip(item_set, item_set))

    rows_to_output_full = ranking_metrics(rec_list, actual_list, method ='bpr')
 
    for row in rows_to_output_full:
        if row['Metric'] == 'Recall' and row['K'] == 50:
            return row['score']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run script parameters')

    parser.add_argument('--dataset', type=str, nargs='?', default='yelp',
                        help='Dataset Name')

    parser.add_argument('--method', type=str, nargs='?', default='BPR',
                        help='method name')

    parser.add_argument('--random_split', type=int, nargs='?', default=1,
                        help='argument to specify random split')

    parser.add_argument('--embed_dim', type=int, nargs='?', default=64,
                        help='embedding dimension')

    parser.add_argument('--batch_size', type=int, nargs='?', default=1024,
                        help='batch size')

    parser.add_argument('--epochs', type=int, nargs='?', default=100,
                        help='# epochs')

    parser.add_argument('--learning_rate', type=float, nargs='?', default=0.0001,
                        help='learning rate')

    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0001,
                        help='L2 reg weight decay')

    parser.add_argument('--lambda_u', type=float, nargs='?', default=0.00000,
                        help='L2 reg hyper-param (users)')

    parser.add_argument('--lambda_i', type=float, nargs='?', default=0.0000,
                        help='L2 reg hyper-param (items)')

    parser.add_argument('--val_iter', type=int, nargs='?', default=1,
                        help='Interval for validation')

    parser.add_argument('--use_gpu', type=int, nargs='?', default=0,
                        help='Flag to use GPU')
    
    parser.add_argument('--social', type=int, nargs='?', default=0,
                        help='Flag to use social')
          
          
    parser.add_argument('--use_pretrain', type=int, nargs='?', default=0,
                        help='Flag to use pretrain')

    args = parser.parse_args()
    SEED = 7
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_ratings, train_set, test_set, num_items, num_users = utils.load_data(args.dataset, args.random_split)

    train_users = [x[0] for x in train_ratings]
    train_users = {x:0 for x in train_users}
    # data loader params
    train_loader_params = {'batch_size': args.batch_size,
                           'shuffle': True,
                           'num_workers': 1}

    test_loader_params = {'batch_size': args.batch_size,
                          'shuffle': False,
                          'num_workers': 1}
    print ("# users", num_users)

    training_set = utils.Dataset(train_ratings)
    train_generator = data.DataLoader(training_set, **train_loader_params)

    checkpoint_path = '../models/{}_{}.pt'.format(args.method, args.dataset)

    model_params = {'num_users': num_users, 'num_items': num_items, 'embed_dim': args.embed_dim, 'method': args.method, 'lambda_u': args.lambda_u, 'lambda_i': args.lambda_i, 'use_gpu': args.use_gpu}
    model = models.BPR(**model_params)
    if args.use_pretrain:
        model.load_state_dict(torch.load(checkpoint_path))
    if args.use_gpu:
        model = model.cuda()

    if args.method == 'BPR':
       optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_performance = 0.0
    for epoch in range(args.epochs):
        # Training
        model.train()
        epoch_loss = 0.
        for user_batch, item_batch in train_generator:
            model.zero_grad()
            optimizer.zero_grad()
            user_batch = torch.LongTensor(user_batch)
            item_batch = torch.LongTensor(item_batch)
            if args.use_gpu:
                user_batch = user_batch.cuda()
                item_batch = item_batch.cuda()
            pos_preds = model(user_batch, item_batch)
            negative_items = model.sample_negs(item_batch, 1) # [B, 1]
            if args.use_gpu:
                negative_items = negative_items.cuda()
            neg_preds = model(user_batch, negative_items.view(-1))
            batch_loss = model.loss(pos_preds, neg_preds)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss
        epoch_loss /= len(train_generator)

        if epoch % args.val_iter == 0:
            model.eval()
            val_performance = evaluate(test_set, train_set, model, use_gpu=args.use_gpu, mode = 'val')
            print ("val performance", val_performance, "best val performance : ", best_val_performance)
            
            if val_performance > best_val_performance:
                # print ("saving best model ... at ", val_performance)
                best_val_performance = val_performance
                torch.save(model.state_dict(), checkpoint_path)

        print ("Epoch {}: Mean Loss: {}".format(epoch, epoch_loss))
        print ("\n\n")
    model = models.BPR(**model_params)
    model.load_state_dict(torch.load(checkpoint_path))
    if args.use_gpu:
        model = model.cuda()
    model.eval()
    print ("Evaluating on test set: ")
    evaluate(test_set, train_set, model, use_gpu=args.use_gpu, mode = 'test')
