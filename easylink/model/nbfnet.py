from collections.abc import Sequence

import torch
from torch import nn
from torch import autograd

from torch_scatter import scatter_add

from torchdrug import core, layers
from torchdrug.layers import functional
from torchdrug.core import Registry as R

import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min

from torchdrug import layers
from torchdrug.layers import functional


# Thanks to origin repo: https://github.com/DeepGraphLearning/NBFNet/

class GeneralizedRelationalConv(layers.MessagePassingBase):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", dependent=True):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)
        if dependent:
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            self.relation = nn.Embedding(num_relation, input_dim)

    def message(self, graph, input):
        assert graph.num_relation == self.num_relation

        batch_size = len(graph.query)
        node_in, node_out, relation = graph.edge_list.t()
        if self.dependent:
            relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation, self.input_dim)
        else:
            relation_input = self.relation.weight.expand(batch_size, -1, -1)
        relation_input = relation_input.transpose(0, 1)
        node_input = input[node_in]
        edge_input = relation_input[relation]

        if self.message_func == "transe":
            message = edge_input + node_input
        elif self.message_func == "distmult":
            message = edge_input * node_input
        elif self.message_func == "rotate":
            node_re, node_im = node_input.chunk(2, dim=-1)
            edge_re, edge_im = edge_input.chunk(2, dim=-1)
            message_re = node_re * edge_re - node_im * edge_im
            message_im = node_re * edge_im + node_im * edge_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        message = torch.cat([message, graph.boundary])

        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1).unsqueeze(-1)
        degree_out = graph.degree_out.unsqueeze(-1).unsqueeze(-1) + 1

        if self.aggregate_func == "sum":
            update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "mean":
            update = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "max":
            update = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
        elif self.aggregate_func == "pna":
            mean = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            sq_mean = scatter_mean(message ** 2 * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            max = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            min = scatter_min(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        return update

    def message_and_aggregate(self, graph, input):
        if graph.requires_grad or self.message_func == "rotate":
            return super(GeneralizedRelationalConv, self).message_and_aggregate(graph, input)

        assert graph.num_relation == self.num_relation

        batch_size = len(graph.query)
        input = input.flatten(1)
        boundary = graph.boundary.flatten(1)

        degree_out = graph.degree_out.unsqueeze(-1) + 1
        if self.dependent:
            relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation, self.input_dim)
            relation_input = relation_input.transpose(0, 1).flatten(1)
        else:
            relation_input = self.relation.weight.repeat(1, batch_size)
        adjacency = graph.adjacency.transpose(0, 1)

        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        if self.aggregate_func == "sum":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            update = update + boundary
        elif self.aggregate_func == "mean":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            update = (update + boundary) / degree_out
        elif self.aggregate_func == "max":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="max", mul=mul)
            update = torch.max(update, boundary)
        elif self.aggregate_func == "pna":
            sum = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            sq_sum = functional.generalized_rspmm(adjacency, relation_input ** 2, input ** 2, sum="add", mul=mul)
            max = functional.generalized_rspmm(adjacency, relation_input, input, sum="max", mul=mul)
            min = functional.generalized_rspmm(adjacency, relation_input, input, sum="min", mul=mul)
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            max = torch.max(max, boundary)
            min = torch.min(min, boundary)
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        return update.view(len(update), batch_size, -1)

    def combine(self, input, update):
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
    
@R.register("model.NBFNet")
class NeuralBellmanFordNetwork(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation=None, symmetric=False,
                 message_func="distmult", aggregate_func="pna", short_cut=False, layer_norm=False, activation="relu",
                 concat_hidden=False, num_mlp_layer=2, dependent=True, remove_one_hop=False,
                 num_beam=10, path_topk=10):
        super(NeuralBellmanFordNetwork, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        if num_relation is None:
            double_relation = 1
        else:
            num_relation = int(num_relation)
            double_relation = num_relation * 2
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.symmetric = symmetric
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.remove_one_hop = remove_one_hop
        self.num_beam = num_beam
        self.path_topk = path_topk

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(GeneralizedRelationalConv(self.dims[i], self.dims[i + 1], double_relation,
                                                               self.dims[0], message_func, aggregate_func, layer_norm,
                                                               activation, dependent))

        feature_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim
        self.query = nn.Embedding(double_relation, input_dim)
        self.mlp = layers.MLP(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [1])

    def remove_easy_edges(self, graph, h_index, t_index, r_index=None):
        if self.remove_one_hop:
            h_index_ext = torch.cat([h_index, t_index], dim=-1)
            t_index_ext = torch.cat([t_index, h_index], dim=-1)
            if r_index is not None:
                any = -torch.ones_like(h_index_ext)
                pattern = torch.stack([h_index_ext, t_index_ext, any], dim=-1)
            else:
                pattern = torch.stack([h_index_ext, t_index_ext], dim=-1)
        else:
            if r_index is not None:
                pattern = torch.stack([h_index, t_index, r_index], dim=-1)
            else:
                pattern = torch.stack([h_index, t_index], dim=-1)
        pattern = pattern.flatten(0, -2)
        edge_index = graph.match(pattern)[0]
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)
        return graph.edge_mask(edge_mask)

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index

    def as_relational_graph(self, graph, self_loop=True):
        # add self loop
        # convert homogeneous graphs to knowledge graphs with 1 relation
        edge_list = graph.edge_list
        edge_weight = graph.edge_weight
        if self_loop:
            node_in = node_out = torch.arange(graph.num_node, device=self.device)
            loop = torch.stack([node_in, node_out], dim=-1)
            edge_list = torch.cat([edge_list, loop])
            edge_weight = torch.cat([edge_weight, torch.ones(graph.num_node, device=self.device)])
        relation = torch.zeros(len(edge_list), 1, dtype=torch.long, device=self.device)
        edge_list = torch.cat([edge_list, relation], dim=-1)
        graph = type(graph)(edge_list, edge_weight=edge_weight, num_node=graph.num_node,
                            num_relation=1, meta_dict=graph.meta_dict, **graph.data_dict)
        return graph

    def bellmanford(self, graph, h_index, r_index, separate_grad=False):
        query = self.query(r_index)
        index = h_index.unsqueeze(-1).expand_as(query)
        boundary = torch.zeros(graph.num_node, *query.shape, device=self.device)
        boundary.scatter_add_(0, index.unsqueeze(0), query.unsqueeze(0))
        with graph.graph():
            graph.query = query
        with graph.node():
            graph.boundary = boundary

        hiddens = []
        step_graphs = []
        layer_input = boundary

        for layer in self.layers:
            if separate_grad:
                step_graph = graph.clone().requires_grad_()
            else:
                step_graph = graph
            hidden = layer(step_graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            step_graphs.append(step_graph)
            layer_input = hidden

        node_query = query.expand(graph.num_node, -1, -1)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "step_graphs": step_graphs,
        }

    def forward(self, graph, h_index, t_index, r_index=None, all_loss=None, metric=None):
        if all_loss is not None:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)

        shape = h_index.shape
        if graph.num_relation:
            graph = graph.undirected(add_inverse=True)
            h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        else:
            graph = self.as_relational_graph(graph)
            h_index = h_index.view(-1, 1)
            t_index = t_index.view(-1, 1)
            r_index = torch.zeros_like(h_index)

        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        output = self.bellmanford(graph, h_index[:, 0], r_index[:, 0])
        feature = output["node_feature"].transpose(0, 1)
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index)

        if self.symmetric:
            assert (t_index[:, [0]] == t_index).all()
            output = self.bellmanford(graph, t_index[:, 0], r_index[:, 0])
            inv_feature = output["node_feature"].transpose(0, 1)
            index = h_index.unsqueeze(-1).expand(-1, -1, inv_feature.shape[-1])
            inv_feature = inv_feature.gather(1, index)
            feature = (feature + inv_feature) / 2

        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)

    def visualize(self, graph, h_index, t_index, r_index):
        assert h_index.numel() == 1 and h_index.ndim == 1
        graph = graph.undirected(add_inverse=True)

        output = self.bellmanford(graph, h_index, r_index, separate_grad=True)
        feature = output["node_feature"]
        step_graphs = output["step_graphs"]

        index = t_index.unsqueeze(0).unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(0, index).squeeze(0)
        score = self.mlp(feature).squeeze(-1)

        edge_weights = [graph.edge_weight for graph in step_graphs]
        edge_grads = autograd.grad(score, edge_weights)
        for graph, edge_grad in zip(step_graphs, edge_grads):
            with graph.edge():
                graph.edge_grad = edge_grad
        distances, back_edges = self.beam_search_distance(step_graphs, h_index, t_index, self.num_beam)
        paths, weights = self.topk_average_length(distances, back_edges, t_index, self.path_topk)

        return paths, weights

    @torch.no_grad()
    def beam_search_distance(self, graphs, h_index, t_index, num_beam=10):
        num_node = graphs[0].num_node
        input = torch.full((num_node, num_beam), float("-inf"), device=self.device)
        input[h_index, 0] = 0

        distances = []
        back_edges = []
        for graph in graphs:
            graph = graph.edge_mask(graph.edge_list[:, 0] != t_index)
            node_in, node_out = graph.edge_list.t()[:2]

            message = input[node_in] + graph.edge_grad.unsqueeze(-1)
            msg_source = graph.edge_list.unsqueeze(1).expand(-1, num_beam, -1)

            is_duplicate = torch.isclose(message.unsqueeze(-1), message.unsqueeze(-2)) & \
                           (msg_source.unsqueeze(-2) == msg_source.unsqueeze(-3)).all(dim=-1)
            is_duplicate = is_duplicate.float() - \
                           torch.arange(num_beam, dtype=torch.float, device=self.device) / (num_beam + 1)
            # pick the first occurrence as the previous state
            prev_rank = is_duplicate.argmax(dim=-1, keepdim=True)
            msg_source = torch.cat([msg_source, prev_rank], dim=-1)

            node_out, order = node_out.sort()
            node_out_set = torch.unique(node_out)
            # sort message w.r.t. node_out
            message = message[order].flatten()
            msg_source = msg_source[order].flatten(0, -2)
            size = scatter_add(torch.ones_like(node_out), node_out, dim_size=num_node)
            msg2out = torch.repeat_interleave(size[node_out_set] * num_beam)
            # deduplicate
            is_duplicate = (msg_source[1:] == msg_source[:-1]).all(dim=-1)
            is_duplicate = torch.cat([torch.zeros(1, dtype=torch.bool, device=self.device), is_duplicate])
            message = message[~is_duplicate]
            msg_source = msg_source[~is_duplicate]
            msg2out = msg2out[~is_duplicate]
            size = scatter_add(torch.ones_like(msg2out), msg2out, dim_size=len(node_out_set))

            if not torch.isinf(message).all():
                distance, rel_index = functional.variadic_topk(message, size, k=num_beam)
                abs_index = rel_index + (size.cumsum(0) - size).unsqueeze(-1)
                back_edge = msg_source[abs_index]
                distance = distance.view(len(node_out_set), num_beam)
                back_edge = back_edge.view(len(node_out_set), num_beam, 4)
                distance = scatter_add(distance, node_out_set, dim=0, dim_size=num_node)
                back_edge = scatter_add(back_edge, node_out_set, dim=0, dim_size=num_node)
            else:
                distance = torch.full((num_node, num_beam), float("-inf"), device=self.device)
                back_edge = torch.zeros(num_node, num_beam, 4, dtype=torch.long, device=self.device)

            distances.append(distance)
            back_edges.append(back_edge)
            input = distance

        return distances, back_edges

    def topk_average_length(self, distances, back_edges, t_index, k=10):
        paths = []
        average_lengths = []

        for i in range(len(distances)):
            distance, order = distances[i][t_index].flatten(0, -1).sort(descending=True)
            back_edge = back_edges[i][t_index].flatten(0, -2)[order]
            for d, (h, t, r, prev_rank) in zip(distance[:k].tolist(), back_edge[:k].tolist()):
                if d == float("-inf"):
                    break
                path = [(h, t, r)]
                for j in range(i - 1, -1, -1):
                    h, t, r, prev_rank = back_edges[j][h, prev_rank].tolist()
                    path.append((h, t, r))
                paths.append(path[::-1])
                average_lengths.append(d / len(path))

        if paths:
            average_lengths, paths = zip(*sorted(zip(average_lengths, paths), reverse=True)[:k])

        return paths, average_lengths