import math
import time
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Batch

from models.loss_functions import recon_loss
from utils.sampler import temporal_sampling
import torch.nn.functional as F


def pointwise_training_epoch(net, optimizer, train_data, full_data, node_features, edge_features, train_rand_sampler, train_neighbor_finder,
               batch_size, num_temporal_hops, n_neighbors, verbose=False):
	num_instance = len(train_data.sources)
	num_batch = math.ceil(num_instance / batch_size)

	net.train()
	batch_loss = []
	train_start_time = time.time()
	for batch_idx in (tqdm(range(0, num_batch)) if verbose else range(0, num_batch)):
		optimizer.zero_grad()
		start_idx = batch_idx * batch_size
		end_idx = min(num_instance, start_idx + batch_size)
		sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
		                                    train_data.destinations[start_idx:end_idx]
		edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
		timestamps_batch = train_data.timestamps[start_idx:end_idx]
		size = len(sources_batch)
		_, negatives_batch = train_rand_sampler.sample(size)

		# sample positive subgraphs
		enclosing_subgraphs_pos = temporal_sampling(sources_batch, destinations_batch,
		                                            timestamps_batch, train_neighbor_finder, train_data,
		                                            node_features, edge_features, full_data.n_unique_nodes,
		                                            num_temporal_hops, n_neighbors)

		# sample negative subgraphs

		enclosing_subgraphs_neg = temporal_sampling(sources_batch, negatives_batch, timestamps_batch,
		                                            train_neighbor_finder, train_data,
		                                            node_features, edge_features, full_data.n_unique_nodes,
		                                            num_temporal_hops, n_neighbors)

		assert len(enclosing_subgraphs_pos) == len(enclosing_subgraphs_neg) == size

		batch_pos = Batch.from_data_list(enclosing_subgraphs_pos)
		h = net(batch_pos)
		target_pos = batch_pos.destination + batch_pos.ptr[:-1]
		pos_edge_proba = net.predict_proba(h[target_pos])

		batch_neg = Batch.from_data_list(enclosing_subgraphs_neg)
		h = net(batch_neg)
		target_neg = batch_neg.destination + batch_neg.ptr[:-1]
		neg_edge_proba = net.predict_proba(h[target_neg])

		loss = recon_loss(pos_edge_proba, neg_edge_proba)
		loss.backward()
		optimizer.step()
		batch_loss.append(loss.item())

	avg_epoch_loss = np.array(batch_loss).mean()
	train_end_time = time.time()
	return avg_epoch_loss, train_end_time-train_start_time

def listwise_training_epoch(net, optimizer, train_data, full_data, node_features, edge_features, train_neighbor_finder,
               batch_size, num_temporal_hops, n_neighbors, verbose=False, coalesce_edges_and_time=False, train_randomize_timestamps=False):


	num_instance = len(train_data.sources)
	num_batch = math.ceil(num_instance / batch_size)

	net.train()
	batch_loss = []
	train_start_time = time.time()
	for batch_idx in (tqdm(range(0, num_batch)) if verbose else range(0, num_batch)):
		optimizer.zero_grad()
		start_idx = batch_idx * batch_size
		end_idx = min(num_instance, start_idx + batch_size)
		sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
		                                    train_data.destinations[start_idx:end_idx]
		edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
		timestamps_batch = train_data.timestamps[start_idx:end_idx]
		size = len(sources_batch)

		# sample positive subgraphs
		enclosing_subgraphs_pos = temporal_sampling(sources_batch, destinations_batch,
		                                            timestamps_batch, train_neighbor_finder, train_data,
		                                            node_features, edge_features, full_data.n_unique_nodes,
		                                            num_temporal_hops, n_neighbors, coalesce_edges_and_time, train_randomize_timestamps)


		batch_pos = Batch.from_data_list(enclosing_subgraphs_pos)
		h = net(batch_pos)
		rank_scores = net.predict_proba(h)
		splits = torch.tensor_split(rank_scores, batch_pos.ptr)

		loss = []
		# TODO : write listwise loss (listnet) in loss_functions.py in model package.
		for sp in splits[1:-1]:
			y = torch.zeros(sp.shape[0], device=net.device)
			# make destination as label 1
			y[1] = 1.0

			# loss.append(-torch.sum(F.softmax(y, dim=0) * F.log_softmax(sp, dim=0)))
			loss.append(-torch.sum(y * F.log_softmax(sp, dim=0)))

		loss = sum(loss) / len(loss)
		loss.backward()
		optimizer.step()
		batch_loss.append(loss.item())

	avg_epoch_loss = np.array(batch_loss).mean()
	train_end_time = time.time()
	return avg_epoch_loss, train_end_time - train_start_time