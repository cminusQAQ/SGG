import os
import tqdm
import json
import base64

from PIL import Image
from io import BytesIO
from importlib_metadata import pathlib

script_dir = os.path.dirname(__file__)
img_root = os.path.join(script_dir, '../../../datasets/VisualGenome')


def generate_img_feat_tsv_and_mapping(filename, img2idx_file):
    img2idx = {}
    idx2img = {}
    all_img_list = sorted(list(pathlib.Path(img_root + '/VG_100K').glob('*')))
    with open(filename, 'w') as out_file:
        for idx, file in tqdm.tqdm(enumerate(all_img_list)):
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
                print(file.name, 'error')
                base64_str = ''
            assert len(base64_str) % 4 == 0
            out_file.write(base64_str + '\n')

    json.dump(img2idx, open(img2idx_file, 'w'))


img_tsv_feat_file = img_root + '/b64_feat.tsv'
mapping_file = img_root + '/img2idx.json'

if not pathlib.Path(img_tsv_feat_file).exists():
    generate_img_feat_tsv_and_mapping(img_tsv_feat_file, mapping_file)
else:
    print(f'Image b64 feature file already exists: {img_tsv_feat_file}')
