import argparse
import logging
import os
import sys
import datetime
from pathlib import Path
import torch
import numpy as np
import random
from utils.data import get_data, RandEdgeSampler
from models.tsar import TSARNet
from utils.sampler import get_neighbor_finder
from utils.helpers import EarlyStopMonitor
from train_eval.training import pointwise_training_epoch
from train_eval.evaluation import pointwise_evaluation

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

def get_checkpoint_path(epoch, time_of_run):
    return f'{args.saved_checkpoints_dir}{args.prefix}-{args.data}-{epoch}-{time_of_run}.pth'

def parse_arguments():
    parser = argparse.ArgumentParser('TSAR Link Prediction Pointwise Training')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Dataset directory')
    parser.add_argument('--data', type=str, default='wikipedia',
                        help='Dataset name (eg. wikipedia, reddit, uci, lastfm etc)')
    parser.add_argument('--prefix', type=str, default='tsar',
                        help='Prefix to name the checkpoints and models')
    parser.add_argument('--train_batch_size', default=64,
                        type=int, help="Train batch size")
    parser.add_argument('--eval_batch_size', default=256, type=int,
                        help="Evaluation batch size (should experiment to make it as big as possible (based on available GPU memory))")
    parser.add_argument('--num_epochs', default=25, type=int,
                        help="Number of training epochs")
    parser.add_argument('--num_layers', default=3, type=int, help="Number of layers")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--emb_dim', type=int, default=128,
                        help="Embedding dimension size")
    parser.add_argument('--time_dim', type=int, default=128,
                        help="Time Embedding dimension size. Give 0 if no time encoding is not to be used")
    parser.add_argument('--num_temporal_hops', type=int, default=3,
                        help="No. of temporal hops for sampling. This should be >= n_layers")
    parser.add_argument('--num_neighbors', type=int, default=5,
                        help="No. of neighbors to sample at each temporal hop")
    parser.add_argument('--uniform_sampling', action='store_true',
                        help='Whether to use uniform sampling for temporal neighbors')
    parser.add_argument('--use_azureml_run_log', action='store_true',
                        help='Whether to use azureml run logging')
    parser.add_argument('--patience', type=int, default=3,
                        help='Patience for early stopping')
    parser.add_argument('--log_dir', type=str, default="logs/",
                        help="directory for logs, specifically for azureml snapshot use logs/")
    parser.add_argument('--saved_models_dir', type=str, default="outputs/saved_models/",
                        help="directory for saved models, specifically for azureml snapshots use ./outputs/saved_models/ ")
    parser.add_argument('--saved_checkpoints_dir', type=str, default="outputs/saved_checkpoints/",
                        help="directory for saved models, specifically for azureml snapshots use ./outputs/saved_checkpoints/ ")
    parser.add_argument('--verbose', type=int, default=0, help="Verbosity 0/1")
    parser.add_argument('--seed', type=int, default=0, help="deterministic seed for training. this is different from that used neighbor finder which uses a local random state")

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args


def run(args):

    # setup seed
    setup_seed(args.seed)
    # set logging
    time_of_run = str(datetime.datetime.now())
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(args.log_dir, 'train_log_{}.log'.format(time_of_run))

    logging.basicConfig(filename=log_file, filemode='w',level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    log = logging.getLogger()
    log.info("Logging set up")



    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    log.info(args)

    Path(args.saved_models_dir).mkdir(parents=True, exist_ok=True)
    model_save_path = os.path.join(args.saved_models_dir, '{}-{}-{}.pth'.format(args.prefix, args.data,time_of_run))

    Path(args.saved_checkpoints_dir).mkdir(parents=True, exist_ok=True)

    early_stopper = EarlyStopMonitor(max_round=args.patience)


    # get data
    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data(args.data_dir, args.data, include_padding=True)

    # setup random samplers
    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
    val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
    test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)

    # In the inductive setting, negatives are sampled only amongst other new nodes
    nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                          seed=1)

    nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                           new_node_test_data.destinations,
                                           seed=3)

    # setup neighbour finder
    train_neighbor_finder = get_neighbor_finder(train_data, uniform=args.uniform_sampling)
    val_neighbor_finder = get_neighbor_finder(full_data, uniform=args.uniform_sampling)
    test_neighbor_finder = get_neighbor_finder(full_data, uniform=args.uniform_sampling)

    # create net and optimizer
    net = TSARNet(emb_dim=args.emb_dim, edge_attr_size=edge_features.shape[1], edge_time_emb_dim=args.time_dim, num_layers=args.num_layers, device=device).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # train, val and test

    for epoch in range(args.num_epochs):

        # training
        loss, total_epoch_time = pointwise_training_epoch(net, optimizer,train_data, full_data, node_features, edge_features, train_rand_sampler, train_neighbor_finder, args.train_batch_size, args.num_temporal_hops, args.num_neighbors, args.verbose)
        log.info('Epoch: {},  loss: {}, time: {:.3f}s'.format(epoch, loss, total_epoch_time))
        # validation
        val_results = pointwise_evaluation(net, val_data, full_data, node_features, edge_features, val_rand_sampler, val_neighbor_finder, args.eval_batch_size, args.num_temporal_hops, args.num_neighbors, args.verbose)
        log.info(
            'Validation ap: {}, auc:{}, time:{}'.format(val_results["ap"], val_results["auc"], val_results["time"]))

        nn_val_results = pointwise_evaluation(net, new_node_val_data, full_data, node_features, edge_features, nn_val_rand_sampler, val_neighbor_finder, args.eval_batch_size, args.num_temporal_hops, args.num_neighbors, args.verbose)
        log.info('New Nodes Validation ap: {}, auc:{}, time:{}'.format(nn_val_results["ap"], nn_val_results["auc"],
                                                                       nn_val_results["time"]))




        # Early stopping
        if early_stopper.early_stop_check(val_results["ap"]):
            log.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
            log.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_model_path = get_checkpoint_path(early_stopper.best_epoch, time_of_run)
            net.load_state_dict(torch.load(best_model_path))
            log.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            net.eval()
            break
        else:
            torch.save(net.state_dict(), get_checkpoint_path(epoch, time_of_run))


    # testing on best loaded model
    test_results = pointwise_evaluation(net, test_data, full_data, node_features, edge_features, test_rand_sampler,
                                       test_neighbor_finder, args.eval_batch_size, args.num_temporal_hops,
                                       args.num_neighbors, args.verbose)
    nn_test_results = pointwise_evaluation(net, new_node_test_data, full_data, node_features, edge_features,
                                          nn_test_rand_sampler, test_neighbor_finder, args.eval_batch_size,
                                          args.num_temporal_hops, args.num_neighbors, args.verbose)

    log.info('Testing ap: {}, auc:{}, time:{}'.format(test_results["ap"], test_results["auc"], test_results["time"]))
    log.info('New Nodes Testing ap: {}, auc:{}, time:{}'.format(nn_test_results["ap"], nn_test_results["auc"],
                                                                   nn_test_results["time"]))



    # save model
    log.info('Saving model')
    torch.save(net.state_dict(), model_save_path)
    log.info('Model saved')



if __name__ == '__main__':
    args  = parse_arguments()
    run(args)
