from fsmodel import building_block as bb

import torch
import subprocess
from collections import OrderedDict


model, img_size = bb.build_model("./model_config.json")

# model map 到 CPU
model = model.cpu()

# 打印模型参数名称
# for name, param in model.named_parameters():
#     print(name)

# 验证函数式前向传播是否正确
fast_weights = OrderedDict(model.named_parameters())
sample_input = torch.rand(1, 3, 84, 84)
logits = model(sample_input)
functional_logits = model.functional_forward(sample_input, fast_weights)
print(logits, functional_logits)

# PyTorch 转 ONNX
onnx_model = "build_model_test.onnx"
onnx_preprocess_model = "build_model_test_preprocess.onnx"

torch.onnx.export(model, sample_input, onnx_model, opset_version=13,
                  input_names=['input'], output_names=['output'])

command = [
    "python",
    "-m",
    "onnxruntime.quantization.preprocess",
    "--input",
    onnx_model,
    "--output",
    onnx_preprocess_model
]

subprocess.run(command, check=True)
