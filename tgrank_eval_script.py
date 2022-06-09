import argparse
import logging
import os
import sys
import datetime
from pathlib import Path
import torch
import numpy as np
from utils.data import get_data
from models.tsar import TSARNet
from utils.sampler import get_neighbor_finder
from train_eval.evaluation import ranking_evaluation


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

def get_checkpoint_path(epoch, time_of_run):
    return f'{args.saved_checkpoints_dir}{args.prefix}-{args.data}-{epoch}-{time_of_run}.pth'

def parse_arguments():
    parser = argparse.ArgumentParser('TSAR Link Prediction Eval')
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

    parser.add_argument('--num_temporal_hops_eval', type=int, default=3,
                        help="No. of temporal hops for sampling. This should be >= n_layers")
    parser.add_argument('--num_neighbors_eval', type=int, default=20,
                        help="No. of neighbors to sample at each temporal hop")
    parser.add_argument('--no_fourier_time_encoding', action='store_true',
                        help='Whether to not use fourier time encoding')
    parser.add_argument('--coalesce_edges_and_time', action='store_true',
                        help='Whether to coalesce edges and time. make sure no_fourier_time_encoding is set and time_dim is 1. else will raise error')

    parser.add_argument('--model_time_of_run', type=str, required=True,
                        help="Time of run of model datetime format")

    parser.add_argument('--num_sample_rank_scores', type=int, default=0,
                        help="No. of candidates to sample in temporal ranking")


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
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(args.log_dir, 'log_eval_temporal_ranking_{}_{}_{}.log'.format(args.prefix, args.data, str(datetime.datetime.now())))

    logging.basicConfig(filename=log_file, filemode='w',level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    log = logging.getLogger()
    log.info("Logging set up")



    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    log.info(args)

    Path(args.saved_models_dir).mkdir(parents=True, exist_ok=True)
    best_model_path = os.path.join(args.saved_models_dir, '{}-{}-{}.pth'.format(args.prefix, args.data, args.model_time_of_run))



    # get data
    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data(args.data_dir, args.data, include_padding=True)


    # setup neighbour finder
    test_neighbor_finder = get_neighbor_finder(full_data, uniform=args.uniform_sampling)

    # create net and optimizer
    net = TSARNet(emb_dim=args.emb_dim, edge_attr_size=edge_features.shape[1], edge_time_emb_dim=args.time_dim, num_layers=args.num_layers, use_fourier_features=not args.no_fourier_time_encoding, device=device).to(device)
    net.load_state_dict(torch.load(best_model_path))
    log.info('Loaded the best model for inference from {}'.format(best_model_path))
    net.eval()

    test_results = ranking_evaluation(net, test_data, full_data, node_features, edge_features,
                                      test_neighbor_finder, args.eval_batch_size, args.num_temporal_hops_eval,
                                      args.num_neighbors_eval, args.verbose, args.coalesce_edges_and_time, args.num_sample_rank_scores)
    log.info('Trasductive Testing time:{}'.format(test_results["time"]))
    log.info(test_results)
    nn_test_results = ranking_evaluation(net, new_node_test_data, full_data, node_features, edge_features,
                                         test_neighbor_finder, args.eval_batch_size,
                                         args.num_temporal_hops_eval, args.num_neighbors_eval, args.verbose, args.coalesce_edges_and_time, args.num_sample_rank_scores)

    log.info('New Nodes Testing time:{}'.format(nn_test_results["time"]))
    log.info(nn_test_results)

    log.info('Done!')

if __name__ == '__main__':
    args  = parse_arguments()
    run(args)
