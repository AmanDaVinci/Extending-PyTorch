# ---
# 01_Matrix-Multiplication.ipynb
# ---

from functools import partial

def test(a, b, compare, compare_name=None):
    if compare_name is None:
        compare_name = compare.__name__
    assert compare(a, b),\
    f"{compare_name} check failed:\n{a}\n{b}"

def test_equality(a, b):
    test(a, b, operator.eq, "Equality")

# TODO: fix this mess
def test_approximately(a, b):
    allclose = partial(torch.allclose, atol=1e-5, rtol=1e-03)
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        a = torch.tensor(a)
        b = torch.tensor(b)
    test(a, b, allclose, "Approximate Equality")

# ---
# 02_Neural-Network-Forward-Pass
# ---

from torchvision.datasets import MNIST

def stats(x):
    shape = f"Shape: {' x '.join(str(dim) for dim in x.shape)}"
    mean = f"Mean: {x.mean():.3f}"
    std = f"Std: {x.std():.3f}"
    print("\n".join([shape,mean,std]))

def normalize(x, mean, std): return (x-mean)/std

# TODO: fix this mess
def get_data():
    dataset = MNIST(root="../data/")
    x, y = dataset.data.float(), dataset.targets
    x_train, x_test = x[:50000], x[50000:]
    y_train, y_test = y[:50000], y[50000:]
    x_train = x_train.reshape(x_train.shape[0],
                              x_train.shape[1]*x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0],
                            x_test.shape[1]*x_test.shape[2])
    mean = x_train.mean()
    std = x_train.std()
    x_train = normalize(x_train, mean, std)
    x_test = normalize(x_test, mean, std)
    return x_train, y_train, x_test, y_test

# ---
# 03_Neural-Network-Backpropagation
# ---

import operator

def test_zero(x, tol=1e-3):
    test(x, tol, operator.le,
         f"Zero (less than tolerance: {tol})")

# ---
# 04_Rebuilding-PyTorch-Essentials
# ---

import torch
from torch.utils.data import DataLoader

def accuracy(input, target):
    return (torch.argmax(input, dim=-1)==target).float().mean()

class Dataset():
    def __init__(self, x, y): self.x, self.y = x, y    
    def __len__(self):        return self.x.shape[0]
    def __getitem__(self, i): return self.x[i], self.y[i]

def get_dataloaders(x_train, y_train, x_test, y_test, batch_size):
    train_ds = Dataset(x_train, y_train)
    test_ds = Dataset(x_test, y_test)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size*2, shuffle=False)
    return train_dl, test_dl

class Databunch:

    def __init__(self, train_dl, test_dl, n_classes=None):
        self.train_dl, self.test_dl = train_dl, test_dl
        self.n_classes = n_classes
    
    @property
    def train_ds(self): return self.train_dl.dataset

    @property
    def test_ds(self):  return self.test_dl.dataset


class Learner():
    def __init__(self, model, data, optimizer, loss_func):
        self.model, self.data = model, data
        self.opt, self.loss_func = optimizer, loss_func

# ---
# 05_Customize-Training-with-Callbacks
# ---

import re

def camel2snake(name):
    _camel_r1 = "(.)([A-Z][a-z]+)"
    _camel_r2 = "([a-z0-9])([A-Z])"
    s1 = re.sub(_camel_r1, r'\1_\2', name)
    return re.sub(_camel_r2, r'\1_\2', s1).lower()


class Callback:
    '''Abstract Callback Class'''
    _order = 0
    
    def set_controller(self, control):  self.control = control
    def __getattr__(self, attr):        return getattr(self.control, attr)
    @property
    def name(self):                     return camel2snake(self.__class__.__name__ or 'callback')
    
    def __call__(self, cb_name):
        cb_func = getattr(self, cb_name, None)
        if cb_func and cb_func(): return True
        return False

class CancelTrainException(Exception): pass
class CancelEpochException(Exception): pass
class CancelBatchException(Exception): pass

class Controller:
    '''Main Controller responsible for callbacks and training loop'''
    
    def __init__(self, callback_list=[]):
        self.cbs = [TrainEval()] + callback_list
        for cb in self.cbs: setattr(self, cb.name, cb)
    
    @property
    def model(self):     return self.learner.model
    @property
    def data(self):      return self.learner.data
    @property
    def opt(self):       return self.learner.opt
    @property
    def loss_func(self): return self.learner.loss_func

    def run_one_batch(self, xb, yb):
        try:
            self.xb, self.yb = xb, yb
            if self('before_batch'): return
            self.pred = self.model(self.xb)
            if self('after_pred'): return
            self.loss = self.loss_func(self.pred, self.yb)
            if self('after_loss') or not self.in_train: return
            self.loss.backward()
            if self('after_backward'): return
            self.opt.step()
            if self('after_step'): return
            self.opt.zero_grad()
        except CancelBatchException: self('after_cancel_batch')
        finally:                     self('after_batch')
        
    def run_all_batches(self, dl):
        self.iters = len(dl)
        try:
            for xb, yb in dl: self.run_one_batch(xb, yb)
        except CancelEpochException: self('after_cancel_epoch')
    
    def train(self, learner, epochs):
        self.learner, self.epochs = learner, epochs
        self.loss = torch.tensor(0.)
        try:
            for cb in self.cbs: 
                cb.set_controller(self)
            if self('before_train'): return
            for epoch in range(self.epochs):
                self.epoch = epoch
                if not self('before_epoch'):
                    self.run_all_batches(self.data.train_dl)
                with torch.no_grad():
                    if not self('before_validate'):
                        self.run_all_batches(self.data.test_dl)
                if self('after_epoch'): break
        except CancelTrainException: self('after_cancel_train') 
        finally:                     self('after_train')
    
    def __call__(self, cb_func_name):
        to_stop = True
        for cb in sorted(self.cbs, key=lambda x: x._order):
            to_stop = cb(cb_func_name) and to_stop
        return to_stop

class Test(Callback):
    '''Stops training after the given number of batches'''
    
    _order = 1
    
    def __init__(self, stop_at=10):
        self.stop_at, self.count = stop_at, 0
    
    def before_batch(self):
        self.count += 1
        print(self.count)
        if self.count >= self.stop_at:
            raise CancelTrainException()
    
    def after_cancel_train(self):
        print(f"Training has been stopped by {self.__class__.__name__} callback")

class TrainEval(Callback):
    """
    Callback to switch between train and eval mode,
    as well as keep track of iterations during training.
    
    Note: This callback is attached by default
    """
    
    def before_train(self):
        self.control.n_epochs = 0.
        self.control.n_iter = 0.
    
    def before_epoch(self):
        self.control.model.train()
        self.control.in_train = True
        self.control.n_epochs = self.control.epoch
    
    def before_batch(self):
        if not self.control.in_train: return
        self.control.n_epochs += 1./self.control.iters
        self.control.n_iter += 1
    
    def before_validate(self):
        self.control.model.eval()
        self.control.in_train = False


class StatsReporter(Callback):
    '''Report training statistics in terms of the given metrics'''
    
    def __init__(self, metrics):
        self.metrics = [] if metrics is None else metrics
            
    def before_epoch(self):
        self.train_loss, self.valid_loss = 0., 0.
        self.train_metrics = torch.tensor([0.]).expand(len(self.metrics))
        self.valid_metrics = torch.tensor([0.]).expand(len(self.metrics))
        self.train_count, self.valid_count = 0, 0
    
    def after_loss(self):
        batch_len = self.control.xb.shape[0]
        if self.control.in_train:
            self.train_count += batch_len
            self.train_loss += self.control.loss*batch_len
            self.train_metrics += torch.tensor([m(self.control.pred,self.control.yb)*batch_len\
                                                for m in self.metrics])
        else:
            self.valid_count += batch_len
            self.valid_loss += self.control.loss*batch_len
            self.valid_metrics += torch.tensor([m(self.control.pred,self.control.yb)*batch_len\
                                                for m in self.metrics])
        
    def after_epoch(self):
        header = f"EPOCH#{self.control.epoch} \t"
        train_avg_loss = self.train_loss / self.train_count
        valid_avg_loss = self.valid_loss / self.valid_count
        train_avg_metrics = self.train_metrics.numpy() / self.train_count
        valid_avg_metrics = self.valid_metrics.numpy() / self.valid_count
        train_str = f"Train loss: {train_avg_loss:.3f} \t metrics: {train_avg_metrics} \t"
        valid_str = f"Valid loss: {valid_avg_loss:.3f} \t metrics: {valid_avg_metrics} \t"
        print(header + train_str + valid_str)


# ---
# 06_Hyperparameter-Scheduling-with-Callbacks
# ---

import matplotlib.pyplot as plt

class Recorder(Callback):
    '''A callback to record the learning rate and loss after each training batch'''
    
    def before_train(self):
        self.lrs = [[] for _ in self.opt.param_groups]
        self.losses = []
    
    def after_batch(self):
        if not self.in_train: return
        for p_group, lr in zip(self.opt.param_groups, self.lrs):
            lr.append(p_group['lr'])
        self.losses.append(self.loss.detach().cpu())
    
    def plot_lrs(self, param_group_id=-1):
        plt.plot(self.lrs[param_group_id])
    
    def plot_losses(self):
        plt.plot(self.losses)
    
    def plot(self, param_group_id=-1, skip_last=0):
        losses = [o.item() for o in self.losses]
        lrs = self.lrs[param_group_id]
        n = len(losses) - skip_last
        plt.xscale('log')
        plt.plot(lrs[:n], losses[:n])
        plt.xlabel('Learning Rates')
        plt.ylabel('Loss in Log Scale')
        plt.title('Learning Rate vs Loss')


class ParamScheduler(Callback):
    """
    Callback for scheduling any hyperparameter
    registered in the state dict of the optimizer
    """
    
    _order = 1
    
    def __init__(self, param_name, sched_funcs):
        self.pname, self.sched_funcs = param_name, sched_funcs
    
    def before_train(self):
        if not isinstance(self.sched_funcs, (list, tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)
    
    def set_param(self):
        assert len(self.opt.param_groups) == len(self.sched_funcs)
        for param_group, func in zip(self.opt.param_groups, self.sched_funcs):
            pos = self.n_epochs/self.epochs
            param_group[self.pname] = func(pos)
    
    def before_batch(self):
        if self.in_train: self.set_param()


def scheduler(f):
    def _inner(start, end): return partial(f, start, end)
    return _inner

@scheduler
def constant(start, end, pos):    return start

@scheduler
def linear(start, end, pos): return start + (end-start)*pos

@scheduler
def exponential(start, end, pos): return start * (end/start)**pos

@scheduler
def cosine(start, end, pos):      return start + (1 + math.cos(math.pi*(1-pos))) * (end-start)/2

def mixed_schedulers(pcts, schedulers):
    assert sum(pcts) == 1
    pcts = torch.tensor([0] + pcts)
    pcts = torch.cumsum(pcts, dim=0)
    def _inner(pos):
        idx = (pos > pcts).nonzero().max()
        lower_bound, upper_bound = pcts[idx], pcts[idx+1]
        relative_pos = (pos - lower_bound) / (upper_bound-lower_bound)
        return schedulers[idx](relative_pos)
    return _inner


class LR_Finder(Callback):
    """
    Finding the best learning rate
    by varying it between reasonable bounds
    and picking the one with the best loss
    """
    
    def __init__(self, min_lr=1e-6, max_lr=10, max_iters=100):
        self.min_lr, self.max_lr, self.max_iters = min_lr, max_lr, max_iters
        self.best_loss = 1e9
    
    def before_batch(self):
        if not self.in_train: return 
        i = self.n_iter/self.max_iters
        lr = self.min_lr * (self.max_lr/self.min_lr)**i
        for group in self.opt.param_groups:
            group['lr'] = lr
    
    def after_step(self):
        if self.n_iter > self.max_iters or self.loss > self.best_loss*10:
            raise CancelTrainException()
        if self.loss < self.best_loss: self.best_loss = self.loss
    
    def after_cancel_train(self):
        print(f"LR Finder found best loss of {self.best_loss:.4f}") 
        print("Plot the 'Learning Rate VS Loss' curve to find the best learning rate")