import os
import tqdm
import json
import base64

from PIL import Image
from io import BytesIO
from importlib_metadata import pathlib

script_dir = os.path.dirname(__file__)
sgg_data_dir = os.path.join(script_dir, '../../../datasets/OFA_data/sgg')
img_root = os.path.join(script_dir, '../../../datasets/VisualGenome')


def generate_img_feat_tsv(filename):
    img2idx = {}
    idx2img = {}
    with open(filename, 'w') as out_file:
        for idx, file in tqdm.tqdm(enumerate(pathlib.Path(img_root + '/VG_100K').glob('*'))):
            img_key = '/VG_100K/' + file.name
            img2idx[img_key] = idx
            idx2img[idx] = img_key

            try:
                img = Image.open(str(file.absolute()))  # path to file
                img_buffer = BytesIO()
                img.save(img_buffer, format=img.format)
                byte_data = img_buffer.getvalue()
                base64_str = base64.b64encode(byte_data)  # bytes
                base64_str = base64_str.decode("utf-8")  # str
            except:
                print(file, 'error')
                base64_str = ''
            out_file.write(base64_str + '\n')

    img2idx_file = img_root + '/img2idx.json'
    json.dump(img2idx, open(img2idx_file, 'w'))


img_tsv_feat_file = img_root + '/b64_feat.tsv'
if not pathlib.Path(img_tsv_feat_file).exists():
    generate_img_feat_tsv(img_tsv_feat_file)
else:
    print(f'Image b64 feature file already exists: {img_tsv_feat_file}')
