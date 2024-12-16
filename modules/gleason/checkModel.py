import glob
import os
from tqdm import tqdm 
from gleasonDiag import main

images = glob.glob(os.path.join('be', 'static', 'upload', 'wsi', '*.png'))
arr = []
for image_path in tqdm(images):
    res = main('wsi', image_path.split('/')[-1])
    # print(image_path, res)
    # res = [1,2,3,4]
    with open(f'modules/gleason/result.txt', 'a', encoding='utf-8') as file:
        file.write(f'{image_path}, {res}' + '\n')
    