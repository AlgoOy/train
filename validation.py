import os
import numpy as np
import torch
import onnx
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms


deploy_dir = "deploy"
torch_model_path = os.path.join(deploy_dir, "fine_tuned_model.pth")
onnx_model_path = os.path.join(deploy_dir, "fine_tuned_model.onnx")
onnx_preprocess_model_path = os.path.join(deploy_dir, "fine_tuned_model_preprocess.onnx")
onnx_preprocess_quantized_model_path = os.path.join(deploy_dir, "fine_tuned_model_preprocess_quantized.onnx")

# 测试数据路径
imgs_path = ["deploy/label_0/image_0.png",
            "deploy/label_1/image_15.png",
            "deploy/label_2/image_30.png",
            "deploy/label_3/image_45.png",
            "deploy/label_4/image_60.png"]
# img_path = "Dataset/R-FSIDS_84_84_RGB/CIC-IDS2017/CIC-IDS2017-Bot/pic-1.png"


def load_image(images_path):
    """Load and preprocess the image."""
    transform = transforms.ToTensor()
    images = []
    for img_path in images_path:
        image = Image.open(img_path)
        images.append(transform(image).unsqueeze(0))
    return torch.cat(images, dim=0)


def validate_onnx_model(model_path, inputs_data):
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    # # 打印可用的执行提供程序
    # available_providers = ort.get_available_providers()
    # print("Available execution providers:", available_providers)

    ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    outputs = []
    # inputs_data --> [N, C, H, W]
    for i in range(len(inputs_data)):
        ort_input = inputs_data[i].numpy()
        ort_input = np.expand_dims(ort_input, axis=0)
        ort_input = {ort_session.get_inputs()[0].name: ort_input}
        output = ort_session.run(None, ort_input)
        outputs.append(output)

    return np.concatenate(outputs, axis=0)


def validate_torch_model(model_path, inputs_data):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()

    with torch.no_grad():
        output = model(inputs_data)
    return output


if __name__ == '__main__':
    inputs_data = load_image(imgs_path)

    ort_outs = validate_onnx_model(onnx_model_path, inputs_data)
    ort_outs_label = np.argmax(ort_outs, axis=-1)
    torch_outs = validate_torch_model(torch_model_path, inputs_data)
    torch_outs_label = torch.argmax(torch_outs, dim=-1)
    train_outs = validate_torch_model("train/_tmm_final.pth", inputs_data)
    train_outs_label = torch.argmax(train_outs, dim=-1)

    print("Output from ONNX model:", ort_outs, ort_outs_label)
    print("Output from Torch model:", torch_outs, torch_outs_label)
    print("Output from Train model:", train_outs, train_outs_label)
    print("Output difference:", np.abs(ort_outs[0] - torch_outs.detach().numpy()).max())