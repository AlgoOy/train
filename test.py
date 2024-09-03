import os
import re
import glob
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm
import subprocess
from onnxruntime .quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

import fsmodel
from fsmodel.dataloader import dataloader_init
from fsmodel.building_block import build_model


train_dir = "train"
deploy_dir = "deploy"
if not os.path.exists(deploy_dir):
    os.makedirs(deploy_dir)

# meta_model = fsmodel.Classifier(3, fsmodel.config.n_way).to(fsmodel.config.device)
meta_model, _ = build_model("./model_config.json")
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

# 载入模型
prefix_origin = "tmm"
prefix = os.path.join(train_dir, prefix_origin)
file = glob.glob(f"{prefix}*.pth")
if file:
    file = file[0]
    max_train_acc = float(re.search(rf"{prefix_origin}_(\d+\.\d+)_\d+\.\d+\.pth", file).group(1))
    max_val_acc = float(re.search(rf"{prefix_origin}_\d+\.\d+_(\d+\.\d+)\.pth", file).group(1))
    meta_model = torch.load(file)
    print(f"Loaded model from {file}")
    print(f"Max train accuracy: {max_train_acc:.3f}")
    print(f"Max validation accuracy: {max_val_acc:.3f}")

# 存储准确率
test_accuracy = []
max_acc = 0

# 模型保存路径
pytorch_model_path = os.path.join(deploy_dir, "fine_tuned_model.pth")
onnx_model_path = os.path.join(deploy_dir, "fine_tuned_model.onnx")
onnx_preprocess_model_path = os.path.join(deploy_dir, "fine_tuned_model_preprocess.onnx")
onnx_preprocess_quantized_model_path = os.path.join(deploy_dir, "fine_tuned_model_preprocess_quantized.onnx")

# 校验数据保存路径
calibration_data_path = os.path.join(deploy_dir, "calibration_data.pth")

# 测试
for _ in tqdm(range(max(1, len(test_loader) // fsmodel.config.test_batches)), desc="Testing"):
    x, test_iter = fsmodel.get_meta_batch(fsmodel.config.test_batches, test_loader, test_iter)
    acc, val_label, support_set, query_set, fast_weights = fsmodel.maml(
        meta_model,
        optimizer,
        x,
        fsmodel.config.n_way,
        fsmodel.config.k_shot,
        fsmodel.config.q_query,
        loss_fn,
        inner_train_step=fsmodel.config.test_inner_train_step,
        inner_lr=fsmodel.config.inner_lr,
        train=False,
        deploy=True
    )
    test_accuracy.append(acc)

    if acc > max_acc:
        max_acc = acc

        with torch.no_grad():
            for param, new_weight in zip(meta_model.parameters(), fast_weights):
                param.copy_(fast_weights[new_weight])

            # 保存微调后的模型
            torch.save(meta_model, pytorch_model_path)

            # 保存 support_set 用作量化校准，不需要保存为图片，方面后续创建DataReader(CalibrationDataReader)
            torch.save(support_set.cpu(), calibration_data_path)

            # 按类别保存 query_set 中的图片
            for i in range(len(val_label)):
                label = val_label[i].item()
                img_dir = os.path.join(deploy_dir, f"label_{label}")
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
                img = query_set[i].cpu().numpy().transpose(1, 2, 0)
                img_path = os.path.join(img_dir, f"image_{i}.png")
                img_rgb = Image.fromarray((img * 255).astype(np.uint8))
                img_rgb.save(img_path)

print(f"Test Accuracy: {np.mean(test_accuracy):.4f}")

model = torch.load(pytorch_model_path, map_location=torch.device('cpu'))
model.eval()

sample_input = torch.rand(1, 3, 84, 84)

y = model(sample_input)

torch.onnx.export(model, sample_input, onnx_model_path, opset_version=13,
                  input_names=['input'], output_names=['output'])

command = [
    "python",
    "-m",
    "onnxruntime.quantization.preprocess",
    "--input",
    onnx_model_path,
    "--output",
    onnx_preprocess_model_path
]

subprocess.run(command, check=True)


class DataReader(CalibrationDataReader):
    def __init__(self, data_path):
        self.data = torch.load(data_path)
        self.index = 0
        self.total_samples = support_set.shape[0]

    def get_next(self):
        if self.index < self.total_samples:
            sample = self.data[self.index].unsqueeze(0).numpy()
            self.index += 1
            return {"input": sample}
        else:
            return None


data_reader = DataReader(calibration_data_path)

quantize_static(
    model_input=onnx_preprocess_model_path,
    model_output=onnx_preprocess_quantized_model_path,
    calibration_data_reader=data_reader,
    # quant_format=QuantFormat.QOperator,
    activation_type=QuantType.QUInt8,
    per_channel=True
)
