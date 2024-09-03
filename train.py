import os
import re
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import fsmodel
from fsmodel.dataloader import dataloader_init
from fsmodel.building_block import build_model


train_dir = "train"
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

# meta_model = fsmodel.Classifier(3, fsmodel.config.n_way).to(fsmodel.config.device)
meta_model = build_model("./model_config.json")[0].to(fsmodel.config.device)
optimizer = torch.optim.Adam(meta_model.parameters(), lr=fsmodel.config.meta_lr)
loss_fn = nn.CrossEntropyLoss().to(fsmodel.config.device)

fsmodel.calculate_model_size(meta_model)

(train_loader, val_loader, test_loader), (train_iter, val_iter, test_iter) = dataloader_init()

# # 确保种子固定，每次运行结果一致
# train = fsmodel.get_meta_batch(train_loader, train_iter)[0][0][0].cpu().numpy()
# train = train.reshape(-1, train.shape[-1])
# np.savetxt("train1.txt", train, fmt='%f')
# val = fsmodel.get_meta_batch(val_loader, val_iter)[0][0][0].cpu().numpy()
# val = val.reshape(-1, val.shape[-1])
# np.savetxt("val1.txt", val, fmt='%f')
# test = fsmodel.get_meta_batch(test_loader, test_iter)[0][0][0].cpu().numpy()
# test = test.reshape(-1, test.shape[-1])
# np.savetxt("test1.txt", test, fmt='%f')

# store accuracy and loss
train_loss = []
train_accuracy = []
val_accuracy = []

max_train_acc = 0
max_val_acc = 0

prefix_origin = "tmm"
prefix = os.path.join(train_dir, prefix_origin)
last_epoch_prefix = "_tmm_final"
last_epoch_prefix = os.path.join(train_dir, last_epoch_prefix)

# # load model from last epoch or best model
# load_model_from_best = False
# file = glob.glob(f"{prefix}*.pth")
# if file:
#     file = file[0]
#     max_train_acc = float(re.search(rf"{prefix_origin}_(\d+\.\d+)_\d+\.\d+\.pth", file).group(1))
#     max_val_acc = float(re.search(rf"{prefix_origin}_\d+\.\d+_(\d+\.\d+)\.pth", file).group(1))
#     if not load_model_from_best:
#         file = glob.glob(f"{last_epoch_prefix}.pth")[0]
#     meta_model = torch.load(file)
#     print(f"Loaded model from {file}")
#     print(f"\tMax train accuracy: {max_train_acc:.3f}\n", f"\tMax validation accuracy: {max_val_acc:.3f}")

for epoch in tqdm(range(fsmodel.config.max_epoch), desc="Epochs"):
    train_meta_loss = []
    train_acc = []

    for step in range(max(1, len(train_loader) // fsmodel.config.meta_batch_size)):
        x, train_iter = fsmodel.get_meta_batch(fsmodel.config.meta_batch_size, train_loader, train_iter)

        meta_loss, acc = fsmodel.maml(
            meta_model,
            optimizer,
            x,
            fsmodel.config.n_way,
            fsmodel.config.k_shot,
            fsmodel.config.q_query,
            loss_fn,
            inner_train_step=fsmodel.config.train_inner_train_step
        )

        train_meta_loss.append(meta_loss.item())
        train_acc.append(acc)

    loss = np.mean(train_meta_loss)
    cur_train_acc = np.mean(train_acc)
    tqdm.write(f"\n\tTrain Loss    : {loss:.3f}")
    tqdm.write(f"\tTrain Accuracy: {cur_train_acc * 100: .3f}%")
    train_loss.append(loss)
    train_accuracy.append(cur_train_acc)

    val_acc = []

    for eval_step in range(max(1, len(val_loader) // fsmodel.config.eval_batches)):
        x, val_iter = fsmodel.get_meta_batch(fsmodel.config.eval_batches, val_loader, val_iter)

        _, acc = fsmodel.maml(
            meta_model,
            optimizer,
            x,
            fsmodel.config.n_way,
            fsmodel.config.k_shot,
            fsmodel.config.q_query,
            loss_fn,
            inner_train_step=fsmodel.config.val_inner_train_step,
            train=False,
        )
        val_acc.append(acc)

    cur_val_acc = np.mean(val_acc)
    tqdm.write(f"\tValidation accuracy: {cur_val_acc * 100: .3f}%")
    val_accuracy.append(cur_val_acc)

    if cur_train_acc > max_train_acc and cur_val_acc > max_val_acc:
        max_train_acc = cur_train_acc
        max_val_acc = cur_val_acc
        file = glob.glob(f"{prefix}*.pth")
        if file:
            os.remove(file[0])
        torch.save(meta_model, f"{prefix}_{max_train_acc:.3f}_{max_val_acc:.3f}.pth")
        tqdm.write(f"Model saved with train accuracy: {max_train_acc:.3f} and validation accuracy: {max_val_acc:.3f}")

# save model
torch.save(meta_model, f"{last_epoch_prefix}.pth")

# save loss and accuracy
np.save(f"{prefix}_loss.npy", train_loss)
np.save(f"{prefix}_train_accuracy.npy", train_accuracy)
np.save(f"{prefix}_val_accuracy.npy", val_accuracy)

# plot loss and accuracy

plt.plot(train_loss, label="Train Loss")
plt.plot(train_accuracy, label="Train Accuracy")
plt.plot(val_accuracy, label="Validation Accuracy")
plt.legend()
plt.show()
