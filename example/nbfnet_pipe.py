import os, sys
import math
import logging
import argparse

import yaml
import jinja2
from jinja2 import meta
import easydict

import torch
from torch.nn import functional as F
from torch.utils import data as torch_data

from torchdrug import core, tasks, metrics, utils, datasets
from torchdrug.utils import comm
from torchdrug.layers import functional
from torchdrug.core import Registry as R

parent_path = os.path.dirname(sys.path[0])
if parent_path not in sys.path:
    sys.path.append(parent_path)
from easylink.model.nbfnet import NeuralBellmanFordNetwork

logger = logging.getLogger(__file__)

@R.register("datasets.CoraLinkPrediction")
class CoraLinkPrediction(datasets.Cora):

    def __init__(self, **kwargs):
        super(CoraLinkPrediction, self).__init__(**kwargs)
        self.transform = None

    def __getitem__(self, index):
        return self.graph.edge_list[index]

    def __len__(self):
        return self.graph.num_edge

    def split(self, ratios=(85, 5, 10)):
        length = self.graph.num_edge
        norm = sum(ratios)
        lengths = [int(r / norm * length) for r in ratios]
        lengths[-1] = length - sum(lengths[:-1])

        g = torch.Generator()
        g.manual_seed(0)
        return torch_data.random_split(self, lengths, generator=g)

@R.register("tasks.LinkPrediction")
class LinkPrediction(tasks.Task, core.Configurable):

    _option_members = ["criterion", "metric"]

    def __init__(self, model, criterion="bce", metric=("auroc", "ap"), num_negative=128, strict_negative=True):
        super(LinkPrediction, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.num_negative = num_negative
        self.strict_negative = strict_negative

    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.num_node = dataset.num_node
        train_mask = train_set.indices
        valid_mask = train_set.indices + valid_set.indices
        train_graph = dataset.graph.edge_mask(train_mask)
        valid_graph = dataset.graph.edge_mask(valid_mask)
        self.register_buffer("train_graph", train_graph.undirected())
        self.register_buffer("valid_graph", valid_graph.undirected())
        self.register_buffer("test_graph", dataset.graph.undirected())

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))

        for criterion, weight in self.criterion.items():
            if criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                neg_weight = torch.ones_like(pred)
                neg_weight[:, 1:] = 1 / self.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)

            loss = loss.mean()
            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    @torch.no_grad()
    def _strict_negative(self, count, split="train"):
        graph = getattr(self, "%s_graph" % split)

        node_in = graph.edge_list[:, 0]
        degree_in = torch.bincount(node_in, minlength=self.num_node)
        prob = (graph.num_node - degree_in - 1).float()

        neg_h_index = functional.multinomial(prob, count, replacement=True)
        any = -torch.ones_like(neg_h_index)
        pattern = torch.stack([neg_h_index, any], dim=-1)
        edge_index, num_t_truth = graph.match(pattern)
        t_truth_index = graph.edge_list[edge_index, 1]
        pos_index = torch.repeat_interleave(num_t_truth)
        t_mask = torch.ones(count, self.num_node, dtype=torch.bool, device=self.device)
        t_mask[pos_index, t_truth_index] = 0
        t_mask.scatter_(1, neg_h_index.unsqueeze(-1), 0)
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        neg_t_index = functional.variadic_sample(neg_t_candidate, num_t_candidate, 1).squeeze(-1)

        return neg_h_index, neg_t_index

    def predict_and_target(self, batch, all_loss=None, metric=None):
        batch_size = len(batch)
        pos_h_index, pos_t_index = batch.t()

        if self.split == "train":
            num_negative = self.num_negative
        else:
            num_negative = 1
        if self.strict_negative or self.split != "train":
            neg_h_index, neg_t_index = self._strict_negative(batch_size * num_negative, self.split)
        else:
            neg_h_index, neg_t_index = torch.randint(self.num_node, (2, batch_size * num_negative), device=self.device)
        neg_h_index = neg_h_index.view(batch_size, num_negative)
        neg_t_index = neg_t_index.view(batch_size, num_negative)
        h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
        t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
        h_index[:, 1:] = neg_h_index
        t_index[:, 1:] = neg_t_index

        pred = self.model(self.train_graph, h_index, t_index, all_loss=all_loss, metric=metric)
        target = torch.zeros_like(pred)
        target[:, 0] = 1
        return pred, target

    def evaluate(self, pred, target):
        pred = pred.flatten()
        target = target.flatten()

        metric = {}
        for _metric in self.metric:
            if _metric == "auroc":
                score = metrics.area_under_roc(pred, target)
            elif _metric == "ap":
                score = metrics.area_under_prc(pred, target)
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric
    
def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    ast = env.parse(raw)
    vars = meta.find_undeclared_variables(ast)
    return vars
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars
def build_solver(cfg, dataset):
    train_set, valid_set, test_set = dataset.split()
    if comm.get_rank() == 0:
        logger.warning(dataset)
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))

    if "fast_test" in cfg:
        if comm.get_rank() == 0:
            logger.warning("Quick test mode on. Only evaluate on %d samples for valid / test." % cfg.fast_test)
        g = torch.Generator()
        g.manual_seed(1024)
        valid_set = torch_data.random_split(valid_set, [cfg.fast_test, len(valid_set) - cfg.fast_test], generator=g)[0]
        test_set = torch_data.random_split(test_set, [cfg.fast_test, len(test_set) - cfg.fast_test], generator=g)[0]
    if hasattr(dataset, "num_relation"):
        cfg.task.model.num_relation = dataset.num_relation

    task = core.Configurable.load_config_dict(cfg.task)
    cfg.optimizer.params = task.parameters()
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)
    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, **cfg.engine)

    if "checkpoint" in cfg:
        solver.load(cfg.checkpoint)

    return solver

def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg

def train_and_validate(cfg, solver):
    if cfg.train.num_epoch == 0:
        return

    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.model.split = "train"
        solver.train(**kwargs)
        solver.save("model_epoch_%d.pth" % solver.epoch)
        solver.model.split = "valid"
        metric = solver.evaluate("valid")
        result = metric[cfg.metric]
        if result > best_result:
            best_result = result
            best_epoch = solver.epoch

    solver.load("model_epoch_%d.pth" % best_epoch)
    return solver

if __name__ == "__main__":
    args, vars = parse_args()
    cfg = load_config(args.config, context=vars)
    print(cfg)
    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = build_solver(cfg, dataset)
    train_and_validate(cfg, solver)