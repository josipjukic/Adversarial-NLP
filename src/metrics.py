import numpy as np
import torch
from tqdm.notebook import tqdm


def accuracy(y_pred, y_gold):
    if y_pred.shape[1] == 1:
        # binary
        preds = torch.round(torch.sigmoid(y_pred)).squeeze()
    else:
        # multi-class
        preds = torch.argmax(y_pred, dim=1)
    correct = (preds == y_gold).float()
    acc = correct.sum() / len(correct)
    return acc.item()


def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # keep any point with a lower cost
            is_efficient[i] = True  # and keep self
    return is_efficient


def init_tqdms(args, iterator):
    epoch_bar = tqdm(desc='Training routine', 
                    total=args.num_epochs,
                    position=0)

    train_bar = tqdm(desc='Train set',
                    total=len(iterator['train']), 
                    position=1)

    val_bar = tqdm(desc='Valid set',
                   total=len(iterator['valid']), 
                   position=1)

    tqdms = dict(main=epoch_bar, train=train_bar, valid=val_bar)
    return tqdms

