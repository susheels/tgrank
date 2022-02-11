# TGRank

## Dataset Links
 
Reddit - http://snap.stanford.edu/jodie/reddit.csv

Wikipedia - http://snap.stanford.edu/jodie/wikipedia.csv

MOOC - http://snap.stanford.edu/jodie/mooc.csv

LastFM - http://snap.stanford.edu/jodie/lastfm.csv

Preprocessed Enron and UCI datasets are taken from - https://github.com/snap-stanford/CAW

If one wants to use raw datasets use the preprocessing code given in `preprocess_data.py` and use it as,

``python preprocess_data.py --data_dir "your raw data dir" --data "name""``

```
usage: Interface for data preprocessing [-h] [--data_dir DATA_DIR] [--data DATA] [--bipartite]
                                        [--num_node_feats NUM_NODE_FEATS]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Dataset Directory
  --data DATA           Dataset name (eg. reddit, wikipedia, mooc, lastfm)
  --bipartite           Whether the graph is bipartite
  --num_node_feats NUM_NODE_FEATS
                        Number of random node features

```
## Usage 

For all parameters of training TGRank follow the commands
```
python tgrank_listwise_train.py -h

```
```
usage: TGRank Interaction Ranking Listwise Training [-h] --data_dir DATA_DIR [--data DATA] [--prefix PREFIX] [--train_batch_size TRAIN_BATCH_SIZE]
                                                    [--eval_batch_size EVAL_BATCH_SIZE] [--num_epochs NUM_EPOCHS] [--num_layers NUM_LAYERS] [--lr LR]
                                                    [--emb_dim EMB_DIM] [--time_dim TIME_DIM] [--num_temporal_hops NUM_TEMPORAL_HOPS] [--num_neighbors NUM_NEIGHBORS]
                                                    [--uniform_sampling] [--patience PATIENCE] [--log_dir LOG_DIR] [--saved_models_dir SAVED_MODELS_DIR]
                                                    [--saved_checkpoints_dir SAVED_CHECKPOINTS_DIR] [--verbose VERBOSE] [--seed SEED]
                                                    [--num_temporal_hops_eval NUM_TEMPORAL_HOPS_EVAL] [--num_neighbors_eval NUM_NEIGHBORS_EVAL]
                                                    [--no_fourier_time_encoding] [--coalesce_edges_and_time] [--train_randomize_timestamps] [--no_id_label]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Dataset directory
  --data DATA           Dataset name (eg. reddit, wikipedia, mooc, lastfm, enron, uci)
  --prefix PREFIX       Prefix to name the checkpoints and models
  --train_batch_size TRAIN_BATCH_SIZE
                        Train batch size
  --eval_batch_size EVAL_BATCH_SIZE
                        Evaluation batch size (should experiment to make it as big as possible (based on available GPU memory))
  --num_epochs NUM_EPOCHS
                        Number of training epochs
  --num_layers NUM_LAYERS
                        Number of layers
  --lr LR
  --emb_dim EMB_DIM     Embedding dimension size
  --time_dim TIME_DIM   Time Embedding dimension size. Give 0 if no time encoding is not to be used
  --num_temporal_hops NUM_TEMPORAL_HOPS
                        No. of temporal hops for sampling candidates during training.
  --num_neighbors NUM_NEIGHBORS
                        No. of neighbors to sample for each candidate node at each temporal hop. This is also the same parameter that samples edges.
  --uniform_sampling    Whether to use uniform sampling for temporal neighbors. Default is most recent sampling.
  --patience PATIENCE   Patience for early stopping
  --log_dir LOG_DIR     directory for storing logs.
  --saved_models_dir SAVED_MODELS_DIR
                        directory for saved models.
  --saved_checkpoints_dir SAVED_CHECKPOINTS_DIR
                        directory for saved checkpoints.
  --verbose VERBOSE     Verbosity 0/1 for tqdm
  --seed SEED           deterministic seed for training. this is different from that by used neighbor finder which uses a local random state
  --num_temporal_hops_eval NUM_TEMPORAL_HOPS_EVAL
                        No. of temporal hops for sampling candidates during evaluation.
  --num_neighbors_eval NUM_NEIGHBORS_EVAL
                        No. of neighbors to sample for each candidate node at each temporal hop during evaluation. This is also the same parameter that samples edges.
  --no_fourier_time_encoding
                        Whether to not use fourier time encoding
  --coalesce_edges_and_time
                        Whether to coalesce edges and time. make sure no_fourier_time_encoding is set and time_dim is 1. else will raise error
  --train_randomize_timestamps
                        Whether to randomize train timestamps i.e. after sampling and before going into TSAR
  --no_id_label         Whether to not use identity label to distinguish source from destinations. Value used to set label diffusion


```

## Example commands
All default parameters are given in the help command.
```
python tgrank_listwise_train.py --data_dir "your data dir" --data enron --prefix tgrank-listwise --verbose 1
python tgrank_listwise_train.py --data_dir "your data dir" --data wikipedia --prefix tgrank-listwise --verbose 1
python tgrank_listwise_train.py --data_dir "your data dir" --data mooc --prefix tgrank-listwise --verbose 1
```
## Examples of running ablations

Here running wikipedia without source specific label diffusion
```
python tgrank_listwise_train.py --data_dir "your data dir" --data wikipedia --prefix tgrank-listwise --verbose 1 --no_id_label
```
Similarly, one can use different parameters like: 

`--coalesce_edges_and_time`, 

`--train_randomize_timestamps`, 

`--no_fourier_time_encoding`

