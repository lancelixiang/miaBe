import torch
import numpy as np
from PIL import Image
import os

import sys
import os.path as osp
rootDir = osp.dirname(osp.dirname(osp.dirname(__file__)))
sys.path.insert(0, rootDir)

from TissueMask import GaussianTissueMask
from SuperpixelExtractor import ColorMergedSuperpixelExtractor
from FeatureExtractor import AugmentedDeepFeatureExtractor
from ViTLike import ViTLike

Image.MAX_IMAGE_PIXELS = 100000000000

tissue_detector = GaussianTissueMask(downsampling_factor=2)
spx_extractor = ColorMergedSuperpixelExtractor(
    superpixel_size=1000,
    max_nr_superpixels=15000,
    blur_kernel_size=3,
    compactness=20,
    threshold=0.01,
    downsampling_factor=4
)


def get_feature_extractor(architecture):
    return AugmentedDeepFeatureExtractor(
        architecture=architecture,
        num_workers=1,
        patch_size=144,
        stride=144,
        resize_size=224,
        batch_size=256,
    )


def main(dir='', img='16B0001851.png'):
    image = np.array(Image.open(os.path.dirname(__file__) +
                     f'/../../be/static/upload/{dir}/' + img))
    tissue_mask = tissue_detector.process(image)
    superpixels, _ = spx_extractor.process(image, tissue_mask)

    mFeatures = get_feature_extractor(
        'MambaVision-S-1K').process(image, superpixels)
    eFeatures = get_feature_extractor(
        'efficientnet_b0').process(image, superpixels)
    features = torch.cat((mFeatures, eFeatures), dim=2).to('cuda')

    model = ViTLike(embed_dim=2048)
    state_dict = torch.load('models/gleason/state_dict.pth', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval().to('cuda')
    labels = model(features)

    return labels[0]


if __name__ == '__main__':
    res = main('202412', '16B0001851.png')
    print('....................res', res)
