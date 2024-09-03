from collections import OrderedDict

import numpy as np
import torch

from .utils import create_label, calculate_accuracy
from . import config as fs_config


def maml(
        model,
        optimizer,
        x,
        n_way,
        k_shot,
        q_query,
        loss_fn,
        inner_train_step=1,
        inner_lr=0.4,
        train=True,
        return_labels=False,
        deploy=False,
        multi_gpus=False
):
    criterion, task_loss, task_acc = loss_fn, [], []
    labels = []

    if multi_gpus:
        functional_forward = model.module.functional_forward
    else:
        functional_forward = model.functional_forward

    for meta_batch in x:
        # Get data
        support_set = meta_batch[: n_way * k_shot]
        query_set = meta_batch[n_way * k_shot:]

        # Copy the params for inner loop
        fast_weights = OrderedDict(model.named_parameters())

        # ---------- INNER TRAIN LOOP ---------- #
        for inner_step in range(inner_train_step):
            # Simply training
            train_label = create_label(fs_config.n_way, fs_config.k_shot).to(fs_config.device)
            logits = functional_forward(support_set, fast_weights)
            loss = criterion(logits, train_label)
            # Inner gradients update! #
            """ Inner Loop Update """
            # TODO: Finish the inner loop update rule
            grads = torch.autograd.grad(loss, list(fast_weights.values()), create_graph=True)
            fast_weights = OrderedDict(
                (name, param - inner_lr * grad) for ((name, param), grad) in zip(fast_weights.items(), grads))
            # raise NotImplementedError
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ #

        # ---------- INNER VALID LOOP ---------- #
        val_label = create_label(n_way, q_query).to(fs_config.device)
        if not return_labels:
            """ training / validation """

            # Collect gradients for outer loop
            logits = functional_forward(query_set, fast_weights)
            loss = criterion(logits, val_label)
            task_loss.append(loss)
            task_acc.append(calculate_accuracy(logits, val_label))
        else:
            """ testing """
            logits = functional_forward(query_set, fast_weights)
            labels.extend(torch.argmax(logits, -1).cpu().numpy())

        # 返回 acc, val_label, query_set, fast_weights
        if deploy:
            if x.shape[0] == 1:
                return task_acc[0], val_label, support_set, query_set, fast_weights
            else:
                raise ValueError("Batch size must be 1 for deployment")

    if return_labels:
        return labels

    task_acc = np.mean(task_acc)

    # Update outer loop
    model.train()
    optimizer.zero_grad()

    meta_batch_loss = torch.stack(task_loss).mean()
    if train:
        """ Outer Loop Update """
        # TODO: Finish the outer loop update
        meta_batch_loss.backward()
        optimizer.step()
        # raise NotimplementedError

    return meta_batch_loss, task_acc
