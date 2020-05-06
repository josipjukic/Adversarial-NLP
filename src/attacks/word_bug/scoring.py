import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import model
import math

import numpy as np
import sys


def replaceone(model, inputs, pred, classes):
    losses = torch.zeros(inputs.size()[0], inputs.size()[1])
    for i in range(inputs.size()[1]):
        tempinputs = inputs.clone()
        tempinputs[:, i] = 2
        with torch.no_grad():
            tempoutput = model(tempinputs)
        losses[:, i] = F.nll_loss(tempoutput, pred, reduce=False)
    return losses


def grad(model, inputs, pred, classes):
    losses1 = torch.zeros(inputs.size()[0], inputs.size()[1])
    dloss = torch.zeros(inputs.size()[0], inputs.size()[1])
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.train()
    embd, output = model(inputs, returnembd=True)
    # embd.retain_grad()
    loss = F.nll_loss(output, pred)

    loss.backward()
    score = (inputs <= 2).float()
    score = -score
    score = embd.grad.norm(2, dim=2) + score * 1e9
    return score


def grad_unconstrained(model, inputs, pred, classes):
    losses1 = torch.zeros(inputs.size()[0], inputs.size()[1])
    dloss = torch.zeros(inputs.size()[0], inputs.size()[1])
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.train()
    embd, output = model(inputs, returnembd=True)
    loss = F.nll_loss(output, pred)

    loss.backward()
    score = embd.grad.norm(2, dim=2)
    return score

def word_target(model, batch, y_preds, num_classes, device):
    inputs = batch[0]
    losses = torch.zeros(inputs.shape)
    target = None
    for i in range(inputs.shape[0]):
        if target:
            index, vals = target
            inputs[i-1,:] = vals
        target = (i, torch.clone(inputs[i,:]))
        inputs[i,:] = 0
        with torch.no_grad():
            out = model.predict_proba(batch)
            losses[i,:] = out.gather(1, y_preds).squeeze()
    
    if target:
        index, vals = target
        inputs[-1,:] = vals
    return 1.-losses


def temporal(model, batch, y_preds, num_classes, device):
    inputs, lengths = batch
    new_preds = torch.zeros(inputs.shape)
    losses = torch.zeros(inputs.shape)
    for i in range(inputs.shape[0]):
        preinputs = inputs[:i+1,:]
        with torch.no_grad():
            new_lengths = torch.min(lengths, torch.tensor(i+1).to(device))
            preout = model.predict_proba((preinputs, new_lengths))
            new_preds[i,:] = preout.gather(1, y_preds).squeeze()
            
    losses[0,:] = new_preds[0,:] - 1.0/num_classes
    for i in range(1, inputs.shape[0]):
        losses[i,:] = new_preds[i,:] - new_preds[i-1,:]

    return losses


def temporal_tail(model, batch, y_preds, num_classes, device):
    inputs, lengths = batch
    new_preds = torch.zeros(inputs.shape)
    losses = torch.zeros(inputs.shape)
    for i in range(inputs.shape[0]):
        postinputs = inputs[i:,:]
        with torch.no_grad():
            new_lengths = torch.max(lengths-i, torch.tensor(1).to(device))
            postout = model.predict_proba((postinputs, new_lengths))
            new_preds[i,:] = postout.gather(1, y_preds).squeeze()
            
    losses[-1,:] = new_preds[-1,:] - 1.0/num_classes
    for i in range(inputs.shape[0]-1):
        losses[i,:] = new_preds[i,:] - new_preds[i+1,:]

    return losses


def combined_temporal(model, batch, y_preds, num_classes, device, alpha=1.):
    temporal_score = temporal(model, batch, y_preds, num_classes, device)
    temporal_tail_score = temporal_tail(model, batch, y_preds, num_classes, device)
    return temporal_score + alpha*temporal_tail_score


def random(inputs, *args, **kwargs):
    losses = torch.rand(inputs.size()[0], inputs.size()[1])
    return losses