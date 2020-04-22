import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.autograd import Variable
import argparse
import model
import math

import numpy as np


import sys
# Input:
# model: the torch model
# input: the input at current stage
#        Torch tensor with size (Batchsize,length)
# Output: score with size (batchsize, length)


def random(model, inputs, pred, classes):
    losses = torch.rand(inputs.size()[0], inputs.size()[1])
    return losses
    # Output a random list


def replaceone(model, inputs, pred, classes):
    losses = torch.zeros(inputs.size()[0], inputs.size()[1])
    for i in range(inputs.size()[1]):
        tempinputs = inputs.clone()
        tempinputs[:, i] = 2
        with torch.no_grad():
            tempoutput = model(tempinputs)
        losses[:, i] = F.nll_loss(tempoutput, pred, reduce=False)
    return losses


def temporal(model, inputs, pred, classes):
    losses1 = torch.zeros(inputs.size()[0], inputs.size()[1])
    dloss = torch.zeros(inputs.size()[0], inputs.size()[1])
    for i in range(inputs.size()[1]):
        tempinputs = inputs[:, :i+1]
        with torch.no_grad():
            tempoutput = torch.exp(model(tempinputs))
        losses1[:, i] = tempoutput.gather(1, pred.view(-1, 1)).view(-1)
    dloss[:, 0] = losses1[:, 0] - 1.0/classes
    for i in range(1, inputs.size()[1]):
        dloss[:, i] = losses1[:, i] - losses1[:, i-1]
    return dloss


def temporaltail(model, inputs, pred, classes):
    losses1 = torch.zeros(inputs.size()[0], inputs.size()[1])
    dloss = torch.zeros(inputs.size()[0], inputs.size()[1])
    for i in range(inputs.size()[1]):
        tempinputs = inputs[:, i:]
        with torch.no_grad():
            tempoutput = torch.exp(model(tempinputs))
        losses1[:, i] = tempoutput.gather(1, pred.view(-1, 1)).view(-1)
    dloss[:, -1] = losses1[:, -1] - 1.0/classes
    for i in range(inputs.size()[1]-1):
        dloss[:, i] = losses1[:, i] - losses1[:, i+1]
    return dloss


def combined(model, inputs, lengths, y_preds, num_classes, device, λ=1.):
    temporal_score = temporal(model, x_in, lengths,
                              y_preds, num_classes, device)
    temporal_tail_score = temporal_tail(
        model, x_in, lengths, y_preds, num_classes, device)
    return temporal_score + *temporal_tail_score


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




def word_target(model, inputs, lengths, y_preds, num_classes, device):
    losses = torch.zeros(inputs.shape)
    target = None
    for i in range(inputs.shape[0]):
        if target:
            index, vals = target
            inputs[i-1,:] = vals
        target = (i, torch.clone(inputs[i,:]))
        inputs[i,:] = 0
        with torch.no_grad():
            out = model.predict_proba(inputs, lengths)
            if num_classes == 2:
                out = torch.cat([1.-out, out], dim=1).to(device)
            losses[i,:] = out.gather(1, y_preds).squeeze()
    
    if target:
        index, vals = target
        inputs[-1,:] = vals
    return 1.-losses


def temporal(model, inputs, lengths, y_preds, num_classes, device):
    new_preds = torch.zeros(inputs.shape)
    losses = torch.zeros(inputs.shape)
    for i in range(inputs.shape[0]):
        preinputs = inputs[:i+1,:]
        with torch.no_grad():
            new_lengths = torch.min(lengths, torch.tensor(i+1).to(device))
            preout = model.predict_proba(preinputs, new_lengths)
            if num_classes == 2:
                preout = torch.cat([1.-preout, preout], dim=1).to(device)
            new_preds[i,:] = preout.gather(1, y_preds).squeeze()
            
    losses[0,:] = new_preds[0,:] - 1.0/num_classes
    for i in range(1, inputs.shape[0]):
        losses[i,:] = new_preds[i,:] - new_preds[i-1,:]

    return losses


def temporal_tail(model, inputs, lengths, y_preds, num_classes, device):
    new_preds = torch.zeros(inputs.shape)
    losses = torch.zeros(inputs.shape)
    for i in range(inputs.shape[0]):
        postinputs = inputs[i:,:]
        with torch.no_grad():
            new_lengths = torch.max(lengths-i, torch.tensor(1).to(device))
            postout = model.predict_proba(postinputs, new_lengths)
            if num_classes == 2:
                postout = torch.cat([1.-postout, postout], dim=1).to(device)
            new_preds[i,:] = postout.gather(1, y_preds).squeeze()
            
    losses[-1,:] = new_preds[-1,:] - 1.0/num_classes
    for i in range(inputs.shape[0]-1):
        losses[i,:] = new_preds[i,:] - new_preds[i+1,:]

    return losses


def combined_temporal(model, inputs, lengths, y_preds, num_classes, device, alpha=1.):
    temporal_score = temporal(model, x_in, lengths, y_preds, num_classes, device)
    temporal_tail_score = temporal_tail(model, x_in, lengths, y_preds, num_classes, device)
    return temporal_score + alpha*temporal_tail_score


def random(inputs, *args, **kwargs):
    losses = torch.rand(inputs.size()[0], inputs.size()[1])
    return losses