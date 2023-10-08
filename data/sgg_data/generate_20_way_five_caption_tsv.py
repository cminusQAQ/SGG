# %%
import os
import sys
import json
import tqdm
import base64
import random
import pathlib

from PIL import Image
from io import BytesIO

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from data.sgg_data.img_feat_tsv import img_tsv_feat_file, mapping_file
from data.tsv_file import TSVFile
from torchvision import transforms

# %%

script_dir = os.path.dirname(__file__)
sgg_data_dir = os.path.join(script_dir, '../../../datasets/OFA_data/sgg')
img_root = os.path.join(script_dir, '../../../datasets/VisualGenome')

data_split = '20_way_caption_five'
split_dir = pathlib.Path(sgg_data_dir) / data_split
vg_dict = json.load((split_dir / 'VG-SGG-dicts-with-attri.json').open())


# %%
num_bins = 1000
max_image_size = 480
patch_image_size = 480

query_template = 'what is the relationship between {}: {} and {}: {}?'
mask_template = 'what is the complete text of "{}: {} <mask> {}: {}"?'

template = query_template
data_prefix = 'query_'

# template = mask_template
# data_prefix = 'mask_'

img2idx = json.load(open(mapping_file))
img_feat_tsv = TSVFile(img_tsv_feat_file, generate_lineidx=True)


def coord2bin(coords, w_resize_ratio, h_resize_ratio):
    # coord_list = [float(coord) for coord in coords.strip().split()]
    coord_list = coords
    bin_list = []
    bin_list += ["<bin_{}>".format(
        int(round(coord_list[0] * w_resize_ratio / max_image_size * (num_bins - 1))))]
    bin_list += ["<bin_{}>".format(
        int(round(coord_list[1] * h_resize_ratio / max_image_size * (num_bins - 1))))]
    bin_list += ["<bin_{}>".format(
        int(round(coord_list[2] * w_resize_ratio / max_image_size * (num_bins - 1))))]
    bin_list += ["<bin_{}>".format(
        int(round(coord_list[3] * h_resize_ratio / max_image_size * (num_bins - 1))))]
    return ' '.join(bin_list)


def get_object(obj):
    *object_coord, object_name = obj.split(';')
    object_coord = [float(x) for x in object_coord]
    return object_name, object_coord


def get_w_h_ratio(img_idx):
    img_str = img_feat_tsv[img_idx][0]
    # print(img_idx, len(img_str))
    img_bytes = base64.urlsafe_b64decode(img_str)
    orig_img = Image.open(BytesIO(img_bytes))
    w, h = orig_img.size
    w_resize_ratio = patch_image_size / w
    h_resize_ratio = patch_image_size / h
    return w_resize_ratio, h_resize_ratio


# %%
pred_set = set()

def collect_train_pos_samples(train_data, train_info, meta_mapping):
    value_lines = []
    image_label_pairs = {}
    for item_id, train_item in enumerate(tqdm.tqdm(train_data)):
        img_id, sample_id = [int(x) for x in train_item['id'].split('_')]

        img_info = train_info[img_id]
        assert img_info['id'] == img_id

        obj_label_ids = img_info['labels']
        obj_label_names = [meta_mapping['idx_to_label'][str(i)] for i in obj_label_ids]
        obj_label_name_str = '&&'.join(obj_label_names)

        rel_label_text = train_item['label']
        sub_info, obj_info, predicate_name = rel_label_text.split('-')
        pred_set.add(predicate_name)
        if predicate_name == 'background':
            continue

        sub, sub_coord = get_object(sub_info)
        obj, obj_coord = get_object(obj_info)

        boxes = img_info['bboxes']
        sub_idx = boxes.index(sub_coord)
        obj_idx = boxes.index(obj_coord)
        assert sub_idx >= 0 and obj_idx >= 0
        if img_id not in image_label_pairs:
            image_label_pairs[img_id] = []
        image_label_pairs[img_id].append((sub_idx, obj_idx))

        img_idx = img2idx[train_item['img_path']]
        w_resize_ratio, h_resize_ratio = get_w_h_ratio(img_idx)
        sub_bins = coord2bin(sub_coord, w_resize_ratio, h_resize_ratio)
        obj_bins = coord2bin(obj_coord, w_resize_ratio, h_resize_ratio)


        instruction = template.format(sub, sub_bins, obj, obj_bins)
        answer = f'1.0|!+{predicate_name}'

        values = [str(item_id), str((img_id, sample_id)), instruction,
                  answer, obj_label_name_str, str(img_idx)]
        value_line = '\t'.join(values) + '\n'
        value_lines.append(value_line)
    return value_lines, image_label_pairs


def collect_train_NA_samples(train_info, img2rel_pairs, meta_mapping):
    no_relation_values = []

    skip_cnt = 0
    skip_img = 0
    for item_id, train_img_item in enumerate(tqdm.tqdm(train_info)):
        img_id = train_img_item['id']
        if img_id not in img2rel_pairs:
            skip_img += 1
            continue
        img_idx = img2idx[train_img_item['image_path']]
        w_resize_ratio, h_resize_ratio = get_w_h_ratio(img_idx)
        boxes = train_img_item['bboxes']

        obj_label_ids = train_img_item['labels']
        obj_label_names = [meta_mapping['idx_to_label'][str(i)] for i in obj_label_ids]
        obj_label_name_str = '&&'.join(obj_label_names)
        assert len(obj_label_names) == len(boxes)

        for sub_idx, (sub_name, sub_box) in enumerate(zip(obj_label_names, boxes)):
            for obj_idx, (obj_name, obj_box) in enumerate(zip(obj_label_names, boxes)):
                if obj_idx == sub_idx:
                    continue
                if (sub_idx, obj_idx) in img2rel_pairs[img_id]:
                    skip_cnt += 1
                    continue

                sub_bins = coord2bin(sub_box, w_resize_ratio, h_resize_ratio)
                obj_bins = coord2bin(obj_box, w_resize_ratio, h_resize_ratio)

                answer = f'1.0|!+no relation'
                instruction = template.format(sub_name, sub_bins, obj_name, obj_bins)

                values = [0, str(item_id), instruction,
                          answer, obj_label_name_str, str(img_idx)]
                no_relation_values.append(values)
    return no_relation_values


# %%
train_rel_data = json.load((split_dir / 'five_caption_train.json').open())
train_info = json.load((split_dir / 'five_caption_info.json').open())
train_rel_lines, train_img2rels = collect_train_pos_samples(train_rel_data, train_info, vg_dict)

# %%
train_NA_lines = collect_train_NA_samples(train_info, train_img2rels, vg_dict)

# %%
ratio = 4
epoch = 50
NA_cnt = int(ratio * len(train_rel_lines))
print(f'collect {NA_cnt} NA samples')

for epoch in tqdm.tqdm(range(epoch)):
    sampled_NA_value_list = random.sample(train_NA_lines, NA_cnt)
    sampled_NA_value_list = [[str(idx), *x[1:]]
                             for idx, x in enumerate(sampled_NA_value_list, len(train_rel_lines))]
    sampled_NA_value_lines = ['\t'.join(x) + '\n' for x in sampled_NA_value_list]
    train_all_lines = train_rel_lines + sampled_NA_value_lines
    random.shuffle(train_all_lines)

    with (split_dir / f'{data_prefix}train_NA{ratio}_E{epoch}.tsv').open('w') as out_file:
        out_file.writelines(train_all_lines)

# %%
