from transformers import AutoModelForImageClassification
from PIL import Image
from timm.data.transforms_factory import create_transform
import torch

import sys
import os.path as osp
parentdir = osp.dirname(osp.dirname(__file__))
sys.path.insert(0, parentdir)

# model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True) # 640
# model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-T2-1K", trust_remote_code=True) # 640
# model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-S-1K", trust_remote_code=True) # 768
model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-B-1K", trust_remote_code=True) # 1024
# model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-L-1K", trust_remote_code=True) # 1568


# prepare image for the model
path = 'demo.jpg'
image = Image.open(path)
input_resolution = (3, 224, 224)  # MambaVision supports any input resolutions
transform = create_transform(input_size=input_resolution,
                             is_training=False,
                             mean=model.config.mean,
                             std=model.config.std,
                             crop_mode=model.config.crop_mode,
                             crop_pct=model.config.crop_pct)
inputs = transform(image).unsqueeze(0).cuda()

# eval mode for inference
model.cuda().eval()

# model inference
outputs = model(inputs)
logits = outputs['logits'] 
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

# feature extra
model = model.model
model.head = torch.nn.Sequential()
features = model(inputs)
print('features', features, features.shape)