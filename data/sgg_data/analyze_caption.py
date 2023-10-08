# %%
import json

sgg_parser = json.load(
    open('../../../datasets/OFA_data/sgg/20_way_caption_five/five_caption_train.json'))
opt = json.load(
    open('../../../datasets/OFA_data/sgg/20_way_caption_five_filtered/opt_5cap_hard_match_20.json'))
pevl = list(map(json.loads, open(
    '../../../datasets/OFA_data/sgg/20_way_caption_five_filtered/pevl_vg_5caption_20.json')))


# %%
def get_triplet(s):
    sub, obj, rel = s.split('-')
    obj = obj.split(';')[-1]
    sub = sub.split(';')[-1]
    return sub, rel, obj


sgg_parser_triplets = [get_triplet(item['label']) for item in sgg_parser]

opt_triplets = sum([[tuple(triplet.split('-')) for triplet in item['triplets']] for item in opt], [])
pevl_triplets = sum([[tuple(data['new_triplet'].split('_'))
                    for data in item['triplet_and_bbox'] if data['result'] is not None]
                    for item in pevl], [])

sgg_parser_triplets = set(sgg_parser_triplets)
opt_triplets = set(opt_triplets)
pevl_triplets = set(pevl_triplets)

# %%
from collections import Counter


opt_sub_counter = Counter([x[0] for x in opt_triplets])
opt_rel_counter = Counter([x[1] for x in opt_triplets])
opt_obj_counter = Counter([x[2] for x in opt_triplets])

pevl_sub_counter = Counter([x[0] for x in pevl_triplets])
pevl_rel_counter = Counter([x[1] for x in pevl_triplets])
pevl_obj_counter = Counter([x[2] for x in pevl_triplets])

sgg_parser_sub_counter = Counter([x[0] for x in sgg_parser_triplets])
sgg_parser_rel_counter = Counter([x[1] for x in sgg_parser_triplets])
sgg_parser_obj_counter = Counter([x[2] for x in sgg_parser_triplets])

# %%
vg_dict = json.load(open('../../../datasets/OFA_data/sgg/20_way/VG-SGG-dicts-with-attri.json'))
idx2rel = vg_dict['idx_to_predicate']
idx2rel = {int(k): v for k, v in idx2rel.items()}

# %%
import numpy
import matplotlib.pyplot as plt

x = numpy.array(range(1, 21))
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)

y_parser = [sgg_parser_rel_counter[idx2rel[p_id]] for p_id in x]
y_opt = [opt_rel_counter[idx2rel[p_id]] for p_id in x]
y_pevl = [pevl_rel_counter[idx2rel[p_id]] for p_id in x]

width = 0.25
ax.bar(x - width, y_parser, width=width, color='b', label='SGG parser')
ax.bar(x, y_opt, width=width, color='g', label='OPT hard match')
ax.bar(x + width, y_pevl, width=width, color='r', label='OPT + pevl')
ax.legend()

ax.set_xticks(x, [idx2rel[p_id] for p_id in x], rotation=45)
plt.show()

# %%

