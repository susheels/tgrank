import numpy as np
import torch
from torch_geometric.data import Data as PyGData
from torch_geometric.utils import coalesce

def convert_to_pyg_edge_tensor(sources, destinations):
    e = np.vstack((sources, destinations))
    return torch.LongTensor(e)

class NeighborFinder:
    def __init__(self, adj_list, uniform=False, seed=None):
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []

        for neighbors in adj_list:
            # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
            # We sort the list based on timestamp
            sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
            self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

        self.uniform = uniform

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def find_before(self, src_idx, cut_time):
        """
        Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

        Returns 3 lists: neighbors, edge_idxs, timestamps

        """
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

        return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[
                                                                                             src_idx][:i]

    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
        """
        Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert (len(source_nodes) == len(timestamps))

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        # NB! All interactions described in these matrices are sorted in each row by time
        neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
        edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                                                     timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

            if len(source_neighbors) > 0 and n_neighbors > 0:
                if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
                    sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

                    neighbors[i, :] = source_neighbors[sampled_idx]
                    edge_times[i, :] = source_edge_times[sampled_idx]
                    edge_idxs[i, :] = source_edge_idxs[sampled_idx]

                    # re-sort based on time
                    pos = edge_times[i, :].argsort()
                    neighbors[i, :] = neighbors[i, :][pos]
                    edge_times[i, :] = edge_times[i, :][pos]
                    edge_idxs[i, :] = edge_idxs[i, :][pos]
                else:
                    # Take most recent interactions
                    source_edge_times = source_edge_times[-n_neighbors:]
                    source_neighbors = source_neighbors[-n_neighbors:]
                    source_edge_idxs = source_edge_idxs[-n_neighbors:]

                    assert (len(source_neighbors) <= n_neighbors)
                    assert (len(source_edge_times) <= n_neighbors)
                    assert (len(source_edge_idxs) <= n_neighbors)

                    neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
                    edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
                    edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

        return neighbors, edge_idxs, edge_times




def get_neighbor_finder(data, uniform, max_node_idx=None):
    max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))

    return NeighborFinder(adj_list, uniform=uniform)


def temporal_sampling(sources_batch, destinations_batch, timestamps_batch, neighbor_finder, data,
                      node_features, edge_features, total_n_unique_nodes, num_temporal_hops=3, n_neighbors=20,
                      coalesce_edges_and_time=False, train_randomize_timestamps=False):
    assert len(sources_batch) == len(destinations_batch) == len(timestamps_batch)
    enclosing_subgraphs = []
    size = len(sources_batch)

    edge_idx_to_data_idx = data.edge_idx_to_data_idx

    data_edges_torch = data.pyg_edge_tensor

    for i in range(size):
        src = sources_batch[i]
        dst = destinations_batch[i]
        cut_off_time = timestamps_batch[i]

        # perform temporal sampling
        visited_nodes = {src, dst}
        fringe = {src, dst}
        nodes = [src, dst]
        visited_edge_idxs = set()
        for h in range(0, num_temporal_hops):
            neighbors, edge_idxs, edge_times = neighbor_finder.get_temporal_neighbor(list(fringe),
                                                                                     len(fringe) * [cut_off_time],
                                                                                     n_neighbors)
            fringe = set(neighbors.flatten()).difference(visited_nodes)
            visited_nodes = visited_nodes.union(fringe)
            visited_edge_idxs = visited_edge_idxs.union(set(edge_idxs.flatten()))
            nodes = nodes + list(fringe)
        visited_edge_idxs = visited_edge_idxs.difference({0})
        if 0 in nodes:
            # 0 is a special token to denote if we didn't find any neighbour node at some point
            nodes.remove(0)

        # len of nodes should be atleast 2 (includes src and dst to begin with)
        assert len(nodes) >= 2

        # get all related info after sampling
        my_data_idxs = []
        for v in visited_edge_idxs:
            my_data_idxs.append(edge_idx_to_data_idx[v])

        ############### old slow way ##########################
        # visited_edge_idxs_mask = np.in1d(data.edge_idxs, np.array(list(visited_edge_idxs)))
        # sampled_edges = data_edges_torch[:, visited_edge_idxs_mask]
        #########################################################

        sampled_edges = data_edges_torch[:, my_data_idxs]

        if coalesce_edges_and_time:
            sampled_edge_features = torch.FloatTensor(edge_features[list(visited_edge_idxs)])
            unit_edge_weights = torch.ones(sampled_edges.shape[1], dtype=torch.float)
            coalesced_edges, (sum_coalesced_edge_features, count_edge_weights) = coalesce(sampled_edges,
                                                                                          [sampled_edge_features,
                                                                                           unit_edge_weights],
                                                                                          reduce='sum')
            mean_coalesced_edge_features = torch.div(sum_coalesced_edge_features, count_edge_weights.view(-1, 1))

            # reindexing node ids to start from 0 - only to be used in message passing
            node_idx = torch.zeros(total_n_unique_nodes + 1, dtype=torch.long)

            # so src and dst always get id 0 and 1 respectively.
            node_idx[torch.LongTensor(nodes)] = torch.arange(len(nodes))

            # reindexing sampled edges to match reindexed node ids
            coalesced_edges = node_idx[coalesced_edges]

            assert coalesced_edges.shape[1] == mean_coalesced_edge_features.shape[0] == count_edge_weights.shape[0]

            pyg_data = PyGData(x=torch.FloatTensor(node_features[nodes]), edge_index=coalesced_edges,
                               edge_attr=mean_coalesced_edge_features, edge_time=count_edge_weights,
                               source=torch.LongTensor([0]),
                               destination=torch.LongTensor([1]))

            pyg_data.num_nodes = len(nodes)
            enclosing_subgraphs.append(pyg_data)


        else:
            # reindexing node ids to start from 0 - only to be used in message passing
            node_idx = torch.zeros(total_n_unique_nodes + 1, dtype=torch.long)

            # so src and dst always get id 0 and 1 respectively.
            node_idx[torch.LongTensor(nodes)] = torch.arange(len(nodes))

            # reindexing sampled edges to match reindexed node ids
            sampled_edges = node_idx[sampled_edges]

            # find timestamp deltas w.r.t cut_off_time
            sampled_edges_times = torch.FloatTensor(cut_off_time - data.timestamps[my_data_idxs])
            if train_randomize_timestamps:
                perm = torch.randperm(sampled_edges_times.shape[0])
                sampled_edges_times = sampled_edges_times[perm]

            # get edge features
            sampled_edge_features = torch.FloatTensor(edge_features[list(visited_edge_idxs)])

            # get node features
            pyg_data = PyGData(x=torch.FloatTensor(node_features[nodes]), edge_index=sampled_edges,
                               edge_attr=sampled_edge_features, edge_time=sampled_edges_times,
                               source=torch.LongTensor([0]),
                               destination=torch.LongTensor([1]))

            pyg_data.num_nodes = len(nodes)
            enclosing_subgraphs.append(pyg_data)
    return enclosing_subgraphs