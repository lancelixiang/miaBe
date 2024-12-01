import os
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from medmnist import INFO
from torchvision import datasets, transforms
import platform


def getPreds(y_score, task=' ', threshold=0.5):
    if task == 'multi-label, binary-class':
        zero = np.zeros_like(y_score)
        one = np.ones_like(y_score)
        y_pre = np.where(y_score < threshold, zero, one)
        return y_pre
    elif task == 'binary-class':
        y_pre = np.zeros(len(y_score))
        for i in range(y_score.shape[0]):
            y_pre[i] = (y_score[i][-1] > threshold)
        return y_pre
    else:
        y_pre = np.zeros(len(y_score))
        for i in range(y_score.shape[0]):
            y_pre[i] = np.argmax(y_score[i])
        return y_pre


class Check(object):
    def __init__(self,
                 key_word: str):
        self.key_word = key_word
        self.separator = '\\' if platform.system() == 'Windows' else '/'

    def __call__(self,
                 file_name: str) -> bool:
        folders = file_name.split(self.separator)
        return folders[-1].find(self.key_word) > -1


def main(fileName, data_flag='pathmnist', gpu_ids='0', batch_size=128, model_path='/../../models/colone/best_model.pth'):
    info = INFO[data_flag]
    task = info['task']
    n_classes = len(info['label'])

    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])

    device = torch.device('cuda:{}'.format(
        gpu_ids[0])) if gpu_ids else torch.device('cpu')

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])])

    test_dataset = datasets.ImageFolder(
        root= os.path.dirname(__file__) + '/../../models/colone/CRC', 
        transform=val_transform, 
        is_valid_file=Check(fileName)
        )

    test_loader = data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for name, param in model.named_parameters():
        if ("bn" not in name):
            param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 500),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(500, n_classes)
    )

    model = model.to(device)
    model.load_state_dict(torch.load(
        os.path.dirname(__file__) + model_path, map_location=device)['net'], strict=True)
    test_metrics = test(model, test_loader, task, device)
    print('predict', test_metrics)
    return test_metrics[0]


def test(model, data_loader, task, device):
    model.eval()
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for _, (inputs, __) in enumerate(data_loader):
            outputs = model(inputs.to(device))
            if task == 'multi-label, binary-class':
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
            y_score = torch.cat((y_score, outputs), 0)

        y_score = y_score.detach().cpu().numpy()
        preds = getPreds(y_score)

        return preds


if __name__ == '__main__':
    main('STR-TCGA-ARGGRLSF.tif')
