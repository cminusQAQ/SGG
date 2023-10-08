
#%%
import tqdm
# KB: caption = 1: 3
# KB, caption, caption, caption

NA_ratio = 1
caption_file_template = '../../../datasets/OFA_data/sgg/20_way_caption_five/query_train_NA{}_E{}.tsv'
caption_file_cnt = 50

KB_file_template = '../../../datasets/OFA_data/sgg/20_way_visualDS/query_train_NA{}_E{}.tsv'
KB_file_cnt = 5


def data_generator(template, cnt):
    while True:
        for E in range(cnt):
            f = template.format(NA_ratio, E)
            with open(f) as data:
                for line in data:
                    line = line.split('\t')
                    line[0] = template[38:46]
                    line = '\t'.join(line)
                    yield line

caption_data_generator = data_generator(caption_file_template, caption_file_cnt)
KB_data_generator = data_generator(KB_file_template, KB_file_cnt)

# %%
n_loop = 10_000
card = 10
card_bsz = 3
KB_repeat = 5
caption_repeat = 5

data = []
for card_idx in range(card):
    for _ in tqdm.tqdm(range(n_loop)):
        for _ in range(caption_repeat):
            for _ in range(card_bsz):
                data.append(next(caption_data_generator))

        for _ in range(KB_repeat):
            for _ in range(card_bsz):
                data.append(next(KB_data_generator))

with open(f'../../../datasets/OFA_data/sgg/20_way_combine/{card}card_card-bsz{card_bsz}_NA{NA_ratio}_KB{KB_repeat}_caption{caption_repeat}_loop{n_loop}.tsv', 'w') as out_file:
    out_file.writelines(data)

# %%
