import math
import torch
import time
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Batch
from sklearn.metrics import average_precision_score, roc_auc_score, ndcg_score
from utils.sampler import temporal_sampling

def pointwise_evaluation(net, eval_data, full_data, node_features, edge_features, eval_rand_sampler, eval_neighbor_finder,
               batch_size, num_temporal_hops, n_neighbors, verbose=False):
	assert eval_rand_sampler.seed is not None
	eval_rand_sampler.reset_random_state()

	ap, auc = [], []
	with torch.no_grad():
		net = net.eval()
		num_instance = len(eval_data.sources)
		num_batch = math.ceil(num_instance / batch_size)

		eval_start_time = time.time()

		for batch_idx in (tqdm(range(0, num_batch)) if verbose else range(0, num_batch)):
			start_idx = batch_idx * batch_size
			end_idx = min(num_instance, start_idx + batch_size)
			sources_batch, destinations_batch = eval_data.sources[start_idx:end_idx], eval_data.destinations[
			                                                                          start_idx:end_idx]
			edge_idxs_batch = eval_data.edge_idxs[start_idx: end_idx]
			timestamps_batch = eval_data.timestamps[start_idx:end_idx]
			size = len(sources_batch)
			_, negatives_batch = eval_rand_sampler.sample(size)

			# sample positive subgraphs
			enclosing_subgraphs_pos = temporal_sampling(sources_batch, destinations_batch,
			                                            timestamps_batch, eval_neighbor_finder, full_data,
			                                            node_features, edge_features, full_data.n_unique_nodes,
			                                            num_temporal_hops, n_neighbors)

			# sample negative subgraphs

			enclosing_subgraphs_neg = temporal_sampling(sources_batch, negatives_batch,
			                                            timestamps_batch, eval_neighbor_finder, full_data,
			                                            node_features, edge_features, full_data.n_unique_nodes,
			                                            num_temporal_hops, n_neighbors)

			batch_pos = Batch.from_data_list(enclosing_subgraphs_pos)
			h = net(batch_pos)
			target_pos = batch_pos.destination + batch_pos.ptr[:-1]
			pos_edge_proba = net.predict_proba(h[target_pos])

			batch_neg = Batch.from_data_list(enclosing_subgraphs_neg)
			h = net(batch_neg)
			target_neg = batch_neg.destination + batch_neg.ptr[:-1]
			neg_edge_proba = net.predict_proba(h[target_neg])

			pred_score = np.concatenate([pos_edge_proba.cpu().numpy(), neg_edge_proba.cpu().numpy()])
			true_label = np.concatenate([np.ones(size), np.zeros(size)])

			ap.append(average_precision_score(true_label, pred_score))
			auc.append(roc_auc_score(true_label, pred_score))

		eval_end_time = time.time()
		return {"ap": np.array(ap).mean(), "auc": np.array(auc).mean(), "time": eval_end_time-eval_start_time}


def ranking_evaluation(net, eval_data, full_data, node_features, edge_features, eval_neighbor_finder,
                       batch_size, num_temporal_hops=3, n_neighbors=20, verbose=False, coalesce_edges_and_time=False,
                       num_sample_rank_scores=0):

	avg_metrics = []
	with torch.no_grad():
		net = net.eval()
		num_instance = len(eval_data.sources)
		num_batch = math.ceil(num_instance / batch_size)
		eval_start_time = time.time()
		for batch_idx in (tqdm(range(0, num_batch)) if verbose else range(0, num_batch)):
			start_idx = batch_idx * batch_size
			end_idx = min(num_instance, start_idx + batch_size)
			sources_batch, destinations_batch = eval_data.sources[start_idx:end_idx], eval_data.destinations[
			                                                                          start_idx:end_idx]
			edge_idxs_batch = eval_data.edge_idxs[start_idx: end_idx]
			timestamps_batch = eval_data.timestamps[start_idx:end_idx]

			# sample positive subgraphs
			enclosing_subgraphs_pos = temporal_sampling(sources_batch, destinations_batch,
			                                            timestamps_batch, eval_neighbor_finder, full_data,
			                                            node_features, edge_features, full_data.n_unique_nodes,
			                                            num_temporal_hops, n_neighbors, coalesce_edges_and_time)

			batch_pos = Batch.from_data_list(enclosing_subgraphs_pos)
			h = net(batch_pos)
			rank_scores = net.predict_proba(h)
			splits = torch.tensor_split(rank_scores, batch_pos.ptr)

			for sp in splits[1:-1]:

				if num_sample_rank_scores:
					true_destination_score = sp[1]
					other_scores = torch.cat([sp[0:1], sp[2:]])
					# sample x number of them if |other_scores| is greater than x
					if other_scores.shape[0] > num_sample_rank_scores:
						indices = torch.randperm(other_scores.shape[0])[:num_sample_rank_scores]

						other_scores = other_scores[indices]

					sp = torch.cat([true_destination_score.view(-1), other_scores])

					argsort = torch.argsort(sp, descending=True)
					# true destination is at position 0
					ranking = (argsort == 0).nonzero()
					ranking = (1 + ranking).cpu().item()

					metrics_dict = {'MRR': 1.0 / ranking,
					                'MR': float(ranking),
					                'HITS@1': 1.0 if ranking <= 1 else 0.0,
					                'HITS@3': 1.0 if ranking <= 3 else 0.0,
					                'HITS@5': 1.0 if ranking <= 5 else 0.0,
					                'HITS@10': 1.0 if ranking <= 10 else 0.0,
					                'HITS@20': 1.0 if ranking <= 20 else 0.0
					                }

					y = np.zeros(sp.shape[0])
					# make destination as label 1
					y[0] = 1.0
					y = np.expand_dims(y, axis=0)
					sp_numpy = sp.cpu().numpy()
					sp_numpy = np.expand_dims(sp_numpy, axis=0)
					assert y.shape == sp_numpy.shape
					ndcg_metrics = {'NDCG@1': ndcg_score(y, sp_numpy, k=1),
					                'NDCG@3': ndcg_score(y, sp_numpy, k=3),
					                'NDCG@5': ndcg_score(y, sp_numpy, k=5),
					                'NDCG@10': ndcg_score(y, sp_numpy, k=10)
					                }
					metrics_dict.update(ndcg_metrics)
					avg_metrics.append(metrics_dict)
				else:

					argsort = torch.argsort(sp, descending=True)
					# true destination is at position 1
					ranking = (argsort == 1).nonzero()
					ranking = (1 + ranking).cpu().item()

					metrics_dict = {'MRR': 1.0 / ranking,
					                'MR': float(ranking),
					                'HITS@1': 1.0 if ranking <= 1 else 0.0,
					                'HITS@3': 1.0 if ranking <= 3 else 0.0,
					                'HITS@5': 1.0 if ranking <= 5 else 0.0,
					                'HITS@10': 1.0 if ranking <= 10 else 0.0,
					                'HITS@20': 1.0 if ranking <= 20 else 0.0
					                }

					y = np.zeros(sp.shape[0])
					# make destination as label 1
					y[1] = 1.0
					y = np.expand_dims(y, axis=0)
					sp_numpy = sp.cpu().numpy()
					sp_numpy = np.expand_dims(sp_numpy, axis=0)
					assert y.shape == sp_numpy.shape
					ndcg_metrics = {'NDCG@1': ndcg_score(y, sp_numpy, k=1),
					                'NDCG@3': ndcg_score(y, sp_numpy, k=3),
					                'NDCG@5': ndcg_score(y, sp_numpy, k=5),
					                'NDCG@10': ndcg_score(y, sp_numpy, k=10)
					                }
					metrics_dict.update(ndcg_metrics)
					avg_metrics.append(metrics_dict)

		final_metrics = {}
		for metric in avg_metrics[0].keys():
			final_metrics[metric] = sum(am[metric]
			                            for am in avg_metrics) / len(avg_metrics)
		eval_end_time = time.time()
		final_metrics["time"] = eval_end_time-eval_start_time
		return final_metrics