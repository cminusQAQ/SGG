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

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from data.sgg_data.img_feat_tsv import img_tsv_feat_file, mapping_file
from data.tsv_file import TSVFile
from torchvision import transforms

# %%

script_dir = os.path.dirname(__file__)
sgg_data_dir = os.path.join(script_dir, '../../../../datasets/OFA_data/sgg')
img_root = os.path.join(script_dir, '../../../../datasets/VisualGenome')

data_split = '20_way_visualDS'
# data_split = '20_way_visualDS_VG'
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
def collect_train_pos_samples(train_data, train_info, meta_mapping):
    value_lines = []
    image_label_pairs = {}
    item_idx = 0
    for item_id, train_item in enumerate(tqdm.tqdm(train_data)):
        img_id, sample_id = [int(x) for x in train_item['id'].split('_')]

        img_info = train_info[img_id]
        assert img_info['id'] == img_id

        obj_label_ids = img_info['labels']
        obj_label_names = [meta_mapping['idx_to_label'][str(i)] for i in obj_label_ids]
        obj_label_name_str = '&&'.join(obj_label_names)

        rel_label_text = train_item['label']
        sub_info, obj_info, predicate_str = rel_label_text.split('-')
        sub, sub_coord = get_object(sub_info)
        obj, obj_coord = get_object(obj_info)

        if predicate_str == 'background':
            continue

        predicate_list = predicate_str.split('%')

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
        for predicate_name in predicate_list:
            if 'background' == predicate_name:
                continue
            answer = f'1.0|!+{predicate_name}'
            values = [str(item_idx), str((img_id, sample_id)), instruction,
                      answer, obj_label_name_str, str(img_idx)]
            value_line = '\t'.join(values) + '\n'
            value_lines.append(value_line)
            item_idx += 1
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
train_rel_data = json.load((split_dir / 'distant_train.json').open())
train_info = json.load((split_dir / 'distant_info.json').open())

# train_rel_data = json.load((split_dir / 'distant_train_vgkb.json').open())
# train_info = json.load((split_dir / 'distant_info_vgkb.json').open())

train_rel_lines, train_img2rels = collect_train_pos_samples(train_rel_data, train_info, vg_dict)

# %%
train_NA_lines = collect_train_NA_samples(train_info, train_img2rels, vg_dict)

# %%
# %%
import json
import torch
import pickle

data_path = '/data_local/yutianyu/OFA/run_scripts/vqa/'

knowledge_prefix = 'OFA_base'
knowledge_prefix = 'optNew_caption_trained_visual_DS'

sample_ids = torch.load(data_path + f'{knowledge_prefix}_sample_ids.pt')
logits = torch.load(data_path + f'{knowledge_prefix}_logits.pt')

train_lines = open('/data_local/yutianyu/datasets/OFA_data/sgg/20_way_visualDS/query_all_train_rel_2316893.tsv').readlines()
ans2label = pickle.load(open('/data_local/yutianyu/datasets/OFA_data/sgg/20_way/20_way_ans2label.pkl', 'rb'))

# %%
vg_dict = json.load(open('/data_local/yutianyu/datasets/OFA_data/sgg/20_way/VG-SGG-dicts-with-attri.json'))
obj_name2idx = vg_dict['label_to_idx']
obj_idx2name = vg_dict['idx_to_label']
rel_name2idx = vg_dict['predicate_to_idx']
rel_idx2name = vg_dict['idx_to_predicate']

# %%
KB = json.load(open('/data_local/yutianyu/datasets/OFA_data/sgg/CCKB.json'))

# %%
import tqdm

soft_label_weight = 0.9

score_idx_pairs = []
for idx, _id in tqdm.tqdm(enumerate(sample_ids)):
    rel_scores = logits[idx]
    rel_line = train_lines[_id.item()]
    annotation_label_str = rel_line.split('\t')[3].split('|!+')[1]
    if annotation_label_str == 'background':
        continue

    obj_pair_name = [x.split()[-1] for x in rel_line.split('\t')[2].split(':')[:2]]
    obj_pair_idx = [str(obj_name2idx[x]) for x in obj_pair_name]
    obj_pair_KB_key = '_'.join(obj_pair_idx)
    obj_pair_KB_value = KB[obj_pair_KB_key]

    selected_logits = [rel_scores[rel_idx] for rel_idx in obj_pair_KB_value]
    selected_scores = torch.tensor(selected_logits).softmax(-1)
    include_NA_logits = selected_logits + [logits[idx][0]]
    include_NA_scores = torch.tensor(include_NA_logits).softmax(-1)

    annotation_label_idx = ans2label[annotation_label_str]

    annotation_label_score = selected_scores[obj_pair_KB_value.index(annotation_label_idx)]
    NA_relative_score = include_NA_scores[-1].item() / (1 / len(include_NA_scores))

    label_conf = float(soft_label_weight * annotation_label_score + (1 - soft_label_weight) / len(selected_scores))
    score_idx_pairs.append((NA_relative_score, label_conf, _id.item()))

# %%
top_k = 100 # of 100, filter samples with high NA prob
assert top_k <= 100

idx_pairs_sort_by_NA_score = sorted(score_idx_pairs, key=lambda x: x[0], reverse=False)

top_k_low_NA_score_idx_list = [x for x in idx_pairs_sort_by_NA_score[:int(len(idx_pairs_sort_by_NA_score) * (top_k / 100))]]
top_k_low_NA_score_lines = [train_lines[x[-1]].replace('1.0|!+', f'{x[1]}|!+')
                            for x in top_k_low_NA_score_idx_list]

# %%
# Fuse seleced samples with NA samples
ratio = 1
epoch = int(100 // top_k)
NA_cnt = int(ratio * len(top_k_low_NA_score_lines))
print(f'collect {NA_cnt} NA samples')

for epoch in tqdm.tqdm(range(epoch)):
    sampled_NA_value_list = random.sample(train_NA_lines, NA_cnt)
    sampled_NA_value_list = [[str(idx), *x[1:]]
                             for idx, x in enumerate(sampled_NA_value_list, len(train_rel_lines))]
    sampled_NA_value_lines = ['\t'.join(x) + '\n' for x in sampled_NA_value_list]
    train_all_lines = top_k_low_NA_score_lines + sampled_NA_value_lines
    random.shuffle(train_all_lines)

    with (split_dir / f'{data_prefix}{knowledge_prefix}-k{top_k}alpha{soft_label_weight}_train_NA{ratio}_E{epoch}.tsv').open('w') as out_file:
        out_file.writelines(train_all_lines)

# %%