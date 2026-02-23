import logging
import torch
import random
import numpy as np
import yaml

def get_config(args,config_file_path):
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        setattr(args, key, value)
    return args

def generate_hidden_dim(hidden_size, z_dim=512):
    hidden_dim = []
    current = z_dim * 4
    for i in range(hidden_size):
        hidden_dim.append(current)
        next_dim = current // 2
        if next_dim < z_dim and hidden_size >= 3:
            next_dim = z_dim
        current = next_dim
    return hidden_dim

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.vals.append(val)
        self.sum = np.sum(self.vals)
        self.count = len(self.vals)
        self.avg = np.mean(self.vals)
        self.std = np.std(self.vals)
        self.min = min(self.vals)
        self.min_ind = self.vals.index(self.min)
        self.max = max(self.vals)
        self.max_ind = self.vals.index(self.max)

def setLogger(logfile):
    logger = logging.getLogger()
    
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console = logging.StreamHandler()
    
    
    while logger.handlers:
        logger.handlers.pop()
    if logfile:
        handler = logging.FileHandler(logfile,mode='w') 
        logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def Init_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_param_groups(model, weight_decay=0.01):
    no_decay_names = model.no_weight_decay()

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(nd_name in name for nd_name in no_decay_names):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

def view_quality_from_pred(pred_list, S_global, mask=None, tau=0.1, eps=1e-9):
    B, C = pred_list[0].shape
    V = len(pred_list)
    pred_stack = torch.stack(pred_list, dim=1)
    quality = []
    for v in range(V):
        p = pred_list[v]
        if mask is not None:
            valid_mask = mask[:, v].bool()
            p = p[valid_mask]
            if p.size(0) == 0:
                quality.append(torch.tensor(-1e9, device=p.device))
                continue

        S_v = p.T @ p
        S_v = S_v / (torch.diag(S_v).unsqueeze(1) + eps)
        S_v.fill_diagonal_(0.)
        S_v = (S_v + S_v.T) / 2
        S_v = S_v / (S_v.sum(1, keepdim=True) + eps)

        diff = torch.norm(S_v - S_global, p="fro")
        q = -diff
        quality.append(q)

    quality = torch.stack(quality)

    if mask is not None:
        mask_bool = mask.bool()
        logits = quality.unsqueeze(0).expand(B, -1) / tau
        logits = logits.masked_fill(~mask_bool, -1e9)
        view_w = torch.softmax(logits, dim=1)
        weights = view_w.unsqueeze(-1)
    else:
        view_w = torch.softmax(quality / tau, dim=0)
        weights = view_w.unsqueeze(0).unsqueeze(-1)

    pred = torch.sum(pred_stack * weights, dim=1)
    return pred, weights.squeeze()