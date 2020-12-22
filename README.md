# Extending Pytorch
---
Extending PyTorch with hooks and a callback system for learning rate policies and gradient health tracking.

The notebooks show how to extend PyTorch using hooks and callbacks.
You can execute the notebooks in the following order:

### Customize training with a callback system
We show how to customize the training routine using callbacks.
Then we build a callback system which can insert arbitary logic at all steps of training.
As an example, we write a callback that reports any given metric during training.

### Hyper-parameter scheduling with callbacks
Here, we use the callback system implemented earlier to schedule learning rates.
Specifically, we build a cyclical learning rate policy with restarts.
We also show how to setup discriminative learning rates for different parameter groups.
In the end, we implement a callback that finds the best learning for us.

### Tracking gradients with hooks
In this notebook, we use PyTorch hooks to look inside the model.
We look at the gradients to track changes in their behaviour during training. 
