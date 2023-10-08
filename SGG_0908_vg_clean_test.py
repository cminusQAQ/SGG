# %%
import random
import json
from collections import defaultdict


train_data = json.load(open('./sgg/50_way/sup_train_50.json'))
val_data = json.load(open('./sgg/50_way/val_data_dedup50.json'))
test_data = json.load(open('./sgg/50_way/test_data_dedup50.json'))

# %%
train_triplet_count = defaultdict(int)
for item in train_data:
    s_raw, o_raw, p = item['label'].split('-')
    s = s_raw.split(';')[-1]
    o = o_raw.split(';')[-1]
    train_triplet_count[(s, p, o)] += 1

print(len(train_triplet_count))

# %%
vg_dict = json.load(open('./sgg/50_way/VG-SGG-dicts-with-attri.json'))
complex_data = val_data + test_data
evaluate_triplet_count = defaultdict(int)

for item in complex_data:
    labels = item['labels']
    for s_idx, o_idx, p_idx in item['gt_triple']:
        s_cls = vg_dict['idx_to_label'][str(labels[s_idx])]
        o_cls = vg_dict['idx_to_label'][str(labels[o_idx])]
        p = vg_dict['idx_to_predicate'][str(p_idx)]
        evaluate_triplet_count[(s_cls, p, o_cls)] += 1
print(len(evaluate_triplet_count))

# %%
freq_th = 1
raw_all_triplets = set(list(train_triplet_count.keys()) + list(evaluate_triplet_count.keys()))
train_triplets = set([k for k, v in train_triplet_count.items() if v >= freq_th ])
evaluate_triplets = set([k for k, v in evaluate_triplet_count.items() if v >= freq_th ])
print(f'Train triplets {len(train_triplet_count)} => {len(train_triplets)}')
print(f'Evaluate triplets {len(evaluate_triplet_count)} => {len(evaluate_triplets)}')

# %%
all_triplets = train_triplets | evaluate_triplets
filtered_triplets = [k for k in raw_all_triplets if k not in all_triplets]
pairname2labels = dict()
for s, p, o in all_triplets:
    if (s, o) not in pairname2labels:
        pairname2labels[(s, o)] = []
    pairname2labels[(s, o)].append(p)

# %%
# Measure Labeling Cost
import tqdm

count = 0

predicate_count = {p: 0 for p in vg_dict['idx_to_predicate'].values()}
subject_count = {s: 0 for s in vg_dict['idx_to_label'].values()}
object_count = {o: 0 for o in vg_dict['idx_to_label'].values()}
triplet_count = {}
pair_count = {}
image_sample_count = defaultdict(int)
images = []

filtered_predicate_count = {p: 0 for p in vg_dict['idx_to_predicate'].values()}
filtered_subject_count = {s: 0 for s in vg_dict['idx_to_label'].values()}
filtered_object_count = {o: 0 for o in vg_dict['idx_to_label'].values()}
filtered_triplet_count = {}
filtered_pair_count = {}
filtered_image_sample_count = defaultdict(int)
filtered_images = []

annotation_instance_list = []
filtered_annotation_instance_list = []

n_image = 0
for test_item_idx, item in tqdm.tqdm(enumerate(test_data)):
    labels = item['labels']
    label_names = [vg_dict['idx_to_label'][str(x)] for x in labels]
    n_samples_in_image = 0
    image_annotation_instance_list = []
    for s_idx, s_cls in enumerate(label_names):
        for o_idx, o_cls in enumerate(label_names):
            if s_idx == o_idx:
                continue
            pairname = (s_cls, o_cls)
            if pairname not in pairname2labels:
                continue

            p_label_list = pairname2labels[pairname]
            n_samples_in_image += len(p_label_list)
            for p in p_label_list:
                image_annotation_instance_list.append((test_item_idx, s_idx, o_idx, vg_dict['predicate_to_idx'][p]))

    count += n_samples_in_image
    n_samples_in_image = n_samples_in_image // 10 * 10
    image_sample_count[n_samples_in_image] += 1

    if n_samples_in_image <= 100:
        images.append(test_item_idx)
        annotation_instance_list += image_annotation_instance_list
        n_image += 1
        for _, s_idx, o_idx, p_idx in image_annotation_instance_list:
            p = vg_dict['idx_to_predicate'][str(p_idx)]
            s_cls = label_names[s_idx]
            o_cls = label_names[o_idx]

            predicate_count[p] += 1
            subject_count[s_cls] += 1
            object_count[o_cls] += 1
            if (s_cls, o_cls) not in pair_count:
                pair_count[(s_cls, o_cls)] = 0
            pair_count[(s_cls, o_cls)] += 1
            if (s_cls, p, o_cls) not in triplet_count:
                triplet_count[(s_cls, p, o_cls)] = 0
            triplet_count[(s_cls, p, o_cls)] += 1
    elif n_samples_in_image <= 2000:
        filtered_images.append(test_item_idx)
        filtered_annotation_instance_list += image_annotation_instance_list
        for _, s_idx, o_idx, p_idx in image_annotation_instance_list:
            p = vg_dict['idx_to_predicate'][str(p_idx)]
            s_cls = label_names[s_idx]
            o_cls = label_names[o_idx]

            filtered_predicate_count[p] += 1
            filtered_subject_count[s_cls] += 1
            filtered_object_count[o_cls] += 1
            if (s_cls, o_cls) not in filtered_pair_count:
                filtered_pair_count[(s_cls, o_cls)] = 0
            filtered_pair_count[(s_cls, o_cls)] += 1
            if (s_cls, p, o_cls) not in filtered_triplet_count:
                filtered_triplet_count[(s_cls, p, o_cls)] = 0
            filtered_triplet_count[(s_cls, p, o_cls)] += 1

print(f'{n_image} images, {len(annotation_instance_list)} annotation instances')

# %%
sorted_sub_count = sorted(list(subject_count.items()), key=lambda x: x[1], reverse=True)
sorted_obj_count = sorted(list(object_count.items()), key=lambda x: x[1], reverse=True)
sorted_pred_count = sorted(list(predicate_count.items()), key=lambda x: x[1], reverse=True)
sorted_pair_count = sorted(list(pair_count.items()), key=lambda x: x[1], reverse=True)
sorted_triplet_count = sorted(list(triplet_count.items()), key=lambda x: x[1], reverse=True)
sorted_image_sample_count = sorted(list(image_sample_count.items()), key=lambda x: x[1], reverse=True)

sorted_filtered_pred_count = sorted(list(filtered_predicate_count.items()), key=lambda x: x[1], reverse=True)
sorted_filtered_pair_count = sorted(list(filtered_pair_count.items()), key=lambda x: x[1], reverse=True)


# %%
import torch

# fine-tuned on training set, with masked encoding strategy
OFA_test_logits = torch.load('/data/private/yutianyu/OFA/run_scripts/vqa/50_way_full_logits.pt')
OFA_test_sample_ids = torch.load('/data/private/yutianyu/OFA/run_scripts/vqa/50_way_full_sample_ids.pt')

# %%
OFA_test_id2logits = {}
for logit, i in zip(OFA_test_logits, OFA_test_sample_ids):
    OFA_test_id2logits[i.item()] = logit

# %%
image_st_ed = json.load(open('./sgg/50_way/mask_test_sample_st_ed_26446.json'))
image_label = json.load(open('./sgg/50_way/mask_test_label_26446.json'))

# %%
def annotation_instance_to_logits(instance):
    test_item_idx, s_idx, o_idx, p_id = instance
    test_item_idx = str(test_item_idx)

    st, ed = image_st_ed[test_item_idx]
    label = image_label[test_item_idx]

    n_box = len(label['boxes'])
    offset = (n_box - 1) * s_idx + o_idx - (o_idx > s_idx)
    logits = OFA_test_id2logits[st + offset]
    return logits.tolist()

annotation_instance_with_logits = [(*instance, annotation_instance_to_logits(instance)) for instance in annotation_instance_list]
# torch.save(annotation_instance_with_logits, './50_way_test_annotation_instances_with_logits.pt')

# %%
import cv2
import numpy
import pathlib
from PIL import Image

def convert_from_cv2_to_image(img: numpy.ndarray) -> Image:
    # return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> numpy.ndarray:
    # return np.asarray(img)
    return cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)


def draw_sample(dirname, sample):
    if not pathlib.Path(dirname).exists():
        pathlib.Path(dirname).mkdir(parents=True)

    img_idx, s_idx, o_idx, p_idx, logits = sample
    img_data = test_data[img_idx]
    img_path = f'/data/private/yutianyu/datasets/VisualGenome/' + img_data['image_path']
    s_box = [int(x) for x in img_data['bboxes'][s_idx]]
    o_box = [int(x) for x in img_data['bboxes'][o_idx]]
    s_cls = vg_dict['idx_to_label'][str(img_data['labels'][s_idx])]
    o_cls = vg_dict['idx_to_label'][str(img_data['labels'][o_idx])]
    p_name = vg_dict['idx_to_predicate'][str(p_idx)]

    img = cv2.imread(img_path)
    img = cv2.rectangle(img, (s_box[0], s_box[1]), (s_box[2], s_box[3]), (0, 0, 255), 2)
    img = cv2.rectangle(img, (o_box[0], o_box[1]), (o_box[2], o_box[3]), (0, 255, 0), 2)

    fname = f'{img_idx}-({s_idx}, {o_idx})-({s_cls}, {p_name}, {o_cls}).jpg'
    cv2.imwrite(f'{dirname}/{fname}', img)

    return img

exit()

# %%
json.dump(annotation_instance_with_logits, open('./construct_clean_vg_test/info.json', 'w'))


n_sharding = 20
sharding_mapping = {i: [] for i in range(n_sharding)}
for idx in range(len(annotation_instance_with_logits)):
    shard_id = random.randint(0, n_sharding - 1)
    sharding_mapping[shard_id].append(idx)
json.dump(sharding_mapping, open('./construct_clean_vg_test/sharding_mapping.json', 'w'))

# %%

for annotation in tqdm.tqdm(annotation_instance_with_logits):
    folder = random.randint(1, 10)
    if not pathlib.Path(f'./construct_clean_vg_test/标注/{folder}').exists():
        pathlib.Path(f'./construct_clean_vg_test/标注/{folder}').mkdir(parents=True)
    draw_sample(f'./construct_clean_vg_test/标注/{folder}', annotation)

exit()
# %%
# Draw samples
import random
th = 0.00
n_sample = 500
n_droped = 0
n_total_viewed = 0

while n_droped < n_sample:
    n_total_viewed += 1
    sample = random.choice(annotation_instance_with_logits)
    sample_logits = sample[-1]
    p_idx = sample[-2]

    sample_scores = torch.softmax(sample_logits, dim=-1)
    no_NA_sample_scores = torch.softmax(sample_logits[1:], dim=-1)
    # score = sample_logits[p_idx]
    score = sample_scores[p_idx]
    # score = no_NA_sample_scores[p_idx - 1]

    if score > th:
        n_droped += 1
        draw_sample(f'./construct_clean_vg_test/标注/th_score={th:.4f}', sample)

print(f'Keep {n_droped / n_total_viewed * 100 :.2f}% samples as {n_droped}')

# %%
