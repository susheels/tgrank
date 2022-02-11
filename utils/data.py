import os
import random
import torch
import numpy as np
import pandas as pd


class Data:
	def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
		self.sources = sources
		self.destinations = destinations
		self.timestamps = timestamps
		self.edge_idxs = edge_idxs
		self.labels = labels
		self.n_interactions = len(sources)
		self.unique_nodes = set(sources) | set(destinations)
		self.n_unique_nodes = len(self.unique_nodes)
		self.edge_idx_to_data_idx = self.make_index()
		self.pyg_edge_tensor = self.make_pyg_edge_tensor()

	def make_index(self):
		edge_idx_to_data_idx = {}
		for i, e_id in enumerate(self.edge_idxs):
			edge_idx_to_data_idx[e_id] = i
		return edge_idx_to_data_idx

	def make_pyg_edge_tensor(self):
		e = np.vstack((self.sources, self.destinations))
		return torch.LongTensor(e)


def get_data(dataset_directory, dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False, include_padding=True):
	# Load data and train val test split
	print(os.path.join(dataset_directory, 'ml_{}.csv'.format(dataset_name)))
	graph_df = pd.read_csv(os.path.join(
		dataset_directory, 'ml_{}.csv'.format(dataset_name)))
	edge_features = np.load(os.path.join(
		dataset_directory, 'ml_{}.npy'.format(dataset_name)))
	node_features = np.load(os.path.join(
		dataset_directory, 'ml_{}_node.npy'.format(dataset_name)))

	if not include_padding:
		print("No padding index")
		graph_df.u -= 1
		graph_df.i -= 1
		graph_df.idx -= 1
		edge_features = edge_features[1:]
		node_features = node_features[1:]

	if randomize_features:
		node_features = np.random.rand(
			node_features.shape[0], node_features.shape[1])

	val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

	sources = graph_df.u.values
	destinations = graph_df.i.values
	edge_idxs = graph_df.idx.values
	labels = graph_df.label.values
	timestamps = graph_df.ts.values

	full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

	random.seed(2020)

	node_set = set(sources) | set(destinations)
	n_total_unique_nodes = len(node_set)

	# Compute nodes which appear at test time
	test_node_set = set(sources[timestamps > val_time]).union(
		set(destinations[timestamps > val_time]))
	# Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
	# their edges from training
	new_test_node_set = set(random.sample(
		test_node_set, int(0.1 * n_total_unique_nodes)))

	# Mask saying for each source and destination whether they are new test nodes
	new_test_source_mask = graph_df.u.map(
		lambda x: x in new_test_node_set).values
	new_test_destination_mask = graph_df.i.map(
		lambda x: x in new_test_node_set).values

	# Mask which is true for edges with both destination and source not being new test nodes (because
	# we want to remove all edges involving any new test node)
	observed_edges_mask = np.logical_and(
		~new_test_source_mask, ~new_test_destination_mask)

	# For train we keep edges happening before the validation time which do not involve any new node
	# used for inductiveness
	train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

	train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
					  edge_idxs[train_mask], labels[train_mask])

	# define the new nodes sets for testing inductiveness of the model
	train_node_set = set(train_data.sources).union(train_data.destinations)
	assert len(train_node_set & new_test_node_set) == 0
	new_node_set = node_set - train_node_set

	val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
	test_mask = timestamps > test_time

	if different_new_nodes_between_val_and_test:
		n_new_nodes = len(new_test_node_set) // 2
		val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
		test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

		edge_contains_new_val_node_mask = np.array(
			[(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
		edge_contains_new_test_node_mask = np.array(
			[(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
		new_node_val_mask = np.logical_and(
			val_mask, edge_contains_new_val_node_mask)
		new_node_test_mask = np.logical_and(
			test_mask, edge_contains_new_test_node_mask)

	else:
		edge_contains_new_node_mask = np.array(
			[(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
		new_node_val_mask = np.logical_and(
			val_mask, edge_contains_new_node_mask)
		new_node_test_mask = np.logical_and(
			test_mask, edge_contains_new_node_mask)

	# validation and test with all edges
	val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
					edge_idxs[val_mask], labels[val_mask])

	test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
					 edge_idxs[test_mask], labels[test_mask])

	# validation and test with edges that at least has one new node (not in training set)
	new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
							 timestamps[new_node_val_mask],
							 edge_idxs[new_node_val_mask], labels[new_node_val_mask])

	new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
							  timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
							  labels[new_node_test_mask])

	print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
																				 full_data.n_unique_nodes))
	print("The training dataset has {} interactions, involving {} different nodes".format(
		train_data.n_interactions, train_data.n_unique_nodes))
	print("The validation dataset has {} interactions, involving {} different nodes".format(
		val_data.n_interactions, val_data.n_unique_nodes))
	print("The test dataset has {} interactions, involving {} different nodes".format(
		test_data.n_interactions, test_data.n_unique_nodes))
	print("The new node validation dataset has {} interactions, involving {} different nodes".format(
		new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
	print("The new node test dataset has {} interactions, involving {} different nodes".format(
		new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
	print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
		len(new_test_node_set)))

	return node_features, edge_features, full_data, train_data, val_data, test_data, \
		new_node_val_data, new_node_test_data


def get_edges_dict(data):
	edges = {}
	for s, d, ts, e_id, l in zip(data.sources, data.destinations, data.timestamps, data.edge_idxs, data.labels):
		tup = (s, d)
		if tup in edges:
			edges[tup].append((ts, e_id, l))
		else:
			edges[tup] = [(ts, e_id, l)]
	return edges


def get_new_edges_split(train_data, eval_data):
	train_edges = get_edges_dict(train_data)
	eval_edges = get_edges_dict(eval_data)
	eval_edges_not_seen = eval_edges.keys() - train_edges.keys()
	sources = []
	destinations = []
	time_stamps = []
	edge_idxs = []
	labels = []
	for k in eval_edges_not_seen:
		for (ts, e_id, l) in eval_edges[k]:
			sources.append(k[0])
			destinations.append(k[1])
			time_stamps.append(ts)
			edge_idxs.append(e_id)
			labels.append(l)
	argsort = np.argsort(time_stamps)
	return Data(np.array(sources)[argsort], np.array(destinations)[argsort], np.array(time_stamps)[argsort], np.array(edge_idxs)[argsort], np.array(labels)[argsort])



class RandEdgeSampler(object):
	def __init__(self, src_list, dst_list, seed=None):
		self.seed = None
		self.src_list = np.unique(src_list)
		self.dst_list = np.unique(dst_list)

		if seed is not None:
			self.seed = seed
			self.random_state = np.random.RandomState(self.seed)

	def sample(self, size):
		if self.seed is None:
			src_index = np.random.randint(0, len(self.src_list), size)
			dst_index = np.random.randint(0, len(self.dst_list), size)
		else:

			src_index = self.random_state.randint(0, len(self.src_list), size)
			dst_index = self.random_state.randint(0, len(self.dst_list), size)
		return self.src_list[src_index], self.dst_list[dst_index]

	def reset_random_state(self):
		self.random_state = np.random.RandomState(self.seed)