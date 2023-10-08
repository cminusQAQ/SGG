# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from io import BytesIO

import os
import json
import pickle
import random
import logging
import warnings

import numpy as np
import torch
import base64
from torchvision import transforms

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset
from data.tsv_file import TSVFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def collate(samples, pad_idx, eos_idx, is_train, parent_task):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    conf = None
    if samples[0].get("conf", None) is not None:
        conf = torch.cat([s['conf'] for s in samples], dim=0)

    ref_dict = None
    if samples[0].get("ref_dict", None) is not None:
        ref_dict = np.array([s['ref_dict'] for s in samples])

    constraint_masks = None
    if samples[0].get("constraint_mask", None) is not None:
        constraint_masks = merge("constraint_mask")

    decoder_prompts = None
    if samples[0].get("decoder_prompt", None) is not None:
        decoder_prompts = np.array([s['decoder_prompt'].tolist() for s in samples])

    prefix_tokens = None
    if samples[0].get("decoder_prompt", None) is not None:
        prefix_tokens = merge("decoder_prompt")
        prefix_tokens = prefix_tokens[:, 1:]

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        )
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "conf": conf,
        "ref_dict": ref_dict,
        "constraint_masks": constraint_masks,
        "decoder_prompts": decoder_prompts,
        "target": target,
        "prefix_tokens": prefix_tokens,
        "KB_label": [s["KB_label"] for s in samples],
        "annotation_label_idx": [s["annotation_label_idx"] for s in samples],
        "answer": [s["answer"] for s in samples]
    }

    return batch


selected_token_mapping = {'no relation': 46776,
                          'carrying': 31123,
                          'covered in': 24914,
                          'covering': 37265,
                          'eating': 29100,
                          'flying in': 28781,
                          'growing on': 11600,
                          'hanging from': 24610,
                          'lying on': 13010,
                          'mounted on': 31034,
                          'painted on': 34843,
                          'parked on': 15129,
                          'playing': 20180,
                          'riding': 23167,
                          'says': 28357,
                          'sitting on': 23428,
                          'standing on': 8190,
                          'using': 10928,
                          'walking in': 11798,
                          'walking on': 36535,
                          'watching': 34464}


# proxy_mapping = {'no relation': 46776,
#                  'carrying': 1500,
#                  'covered in': 1501,
#                  'covering': 1502,
#                  'eating': 1503,
#                  'flying in': 1504,
#                  'growing on': 1505,
#                  'hanging from': 1506,
#                  'lying on': 1507,
#                  'mounted on': 1508,
#                  'painted on': 1509,
#                  'parked on': 1510,
#                  'playing': 1511,
#                  'riding': 1512,
#                  'says': 1513,
#                  'sitting on': 1514,
#                  'standing on': 1515,
#                  'using': 1516,
#                  'walking in': 1517,
#                  'walking on': 1518,
#                  'watching': 1519}

def mask_sub(question_str: str):
    assert question_str.startswith('what is the relationship')
    sub_name_st_idx = question_str.index('between ') + 8
    sub_name_ed_idx = question_str.index(':')
    prev_sub = question_str[:sub_name_st_idx]
    post_sub = question_str[sub_name_ed_idx:]
    out = prev_sub + 'mask' + post_sub
    return out


def mask_obj(question_str):
    assert question_str.startswith('what is the relationship')
    obj_name_st_idx = question_str.index('> and ') + 6
    obj_name_ed_idx = question_str.index(':', obj_name_st_idx)
    prev_obj = question_str[:obj_name_st_idx]
    post_obj = question_str[obj_name_ed_idx:]
    out = prev_obj + 'mask' + post_obj
    return out


class VqaGenDataset(OFADataset):
    def __init__(
            self,
            split,
            dataset,
            bpe,
            src_dict,
            tgt_dict=None,
            max_src_length=128,
            max_object_length=30,
            max_tgt_length=30,
            patch_image_size=224,
            add_object=False,
            constraint_trie=None,
            imagenet_default_mean_and_std=False,
            prompt_type="none",
            parent_task_ref=None,
            label_proxy='answer'
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_object_length = max_object_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size

        self.add_object = add_object
        self.constraint_trie = constraint_trie
        self.prompt_type = prompt_type
        self.label_proxy = label_proxy

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.img_feat_file = TSVFile(
            os.path.join(os.path.dirname(__file__), '../../../datasets/VisualGenome/b64_feat.tsv'))
        self.coco_2017_img_feat_file = TSVFile(
            os.path.join(os.path.dirname(__file__), '../../../datasets/COCO/b64_feat.tsv'))
        self.parent_task_ref = parent_task_ref
        self.KB = json.load(open('../../../datasets/OFA_data/sgg/CCKB.json'))
        self.ans2label = pickle.load(open('../../../datasets/OFA_data/sgg/20_way_caption_five_filtered/20_way_ans2label.pkl', 'rb'))
        # self.ans2label = pickle.load(open('../../../datasets/OFA_data/sgg/50_way/50_way_ans2label.pkl', 'rb'))
        vg_dict = json.load(open('../../../datasets/OFA_data/sgg/20_way_caption_five_filtered/VG-SGG-dicts-with-attri.json'))
        # vg_dict = json.load(open('../../../datasets/OFA_data/sgg/50_way/VG-SGG-dicts-with-attri.json'))
        self.obj_name2idx = vg_dict['label_to_idx']


    def __getitem__(self, index):
        item = self.dataset[index]
        if len(item) == 5:
            uniq_id, image, question, ref, predict_objects = item
        else:
            uniq_id, image, question, ref, predict_objects, caption = item

        if uniq_id == 'coco':
            image = Image.open(BytesIO(base64.urlsafe_b64decode(self.coco_2017_img_feat_file[int(image)][0])))
        else:
            image = Image.open(BytesIO(base64.urlsafe_b64decode(self.img_feat_file[int(image)][0])))
        # image = Image.open(image)
        # image = Image.open(image.replace('/data_local', '/data2/private'))
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])
        obj_pair_name = [x.split()[-1] for x in question.split(':')[:2]]
        # obj_pair_name[0] = obj_pair_name[0][1:] # for mask template such as what is the complete text of "handle: <bin_283> <bin_864> <bin_755> <bin_993> <mask> door: <bin_0> <bin_0> <bin_878> <bin_996>"?
        # print('Object names:', obj_pair_name)

        # Mask obj name by probability
        mask_rate = 0 / 100
        is_train = 'train' in self.dataset.file_path
        if is_train and random.random() < mask_rate:
            question = mask_sub(question)
        if is_train and random.random() < mask_rate:
            question = mask_obj(question)

        question = self.pre_question(question, self.max_src_length)
        question = question + '?' if not question.endswith('?') else question
        src_item = self.encode_text(' {}'.format(question))

        def map_alias(name):
            if name == 'sit on': return 'sitting on'
            elif name == 'grow on': return 'growing on'
            elif name == 'eat': return 'eating'
            elif name == 'lie on': return 'lying on'
            elif name == 'cover': return 'covering'
            elif name == 'hang from': return 'hanging from'
            elif name in self.ans2label:
                return name
            else:
                print(f'@@@@ ERROR IN DATA @@@@ {name}')
                return 'no relation'
        ref_dict = {map_alias(item.split('|!+')[1]): float(item.split('|!+')[0]) for item in ref.split('&&')}

        answer = max(ref_dict, key=ref_dict.get)
        conf = torch.tensor([ref_dict[answer]])
        if self.label_proxy == 'answer':
            tgt_item = self.encode_text(" {}".format(answer))
        elif self.label_proxy == 'selected_token':
            assert False
            tgt_item = torch.tensor([selected_token_mapping[answer]]).long()
        else:
            raise NotImplementedError(f'label_proxy {self.label_proxy} not implemented yet.')

        if self.add_object and predict_objects is not None:
            predict_object_seq = ' '.join(predict_objects.strip().split('&&')[:self.max_object_length])
            predict_object_item = self.encode_text(" object: {}".format(predict_object_seq))
            src_item = torch.cat([src_item, predict_object_item])

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        if self.prompt_type == 'none':
            prev_output_item = torch.cat([self.bos_item, tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = self.bos_item
        elif self.prompt_type == 'src':
            prev_output_item = torch.cat([src_item, tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = src_item
        elif self.prompt_type == 'prev_output':
            prev_output_item = torch.cat([src_item[:-1], tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = src_item[:-1]
        else:
            raise NotImplementedError
        target_item[:-len(tgt_item) - 1] = self.tgt_dict.pad()

        obj_pair_idx = [str(self.obj_name2idx[x]) for x in obj_pair_name]
        obj_pair_KB_key = '_'.join(obj_pair_idx)

        # TODO: assure all pos pairs in KB and NA pairs not in KB
        # which not holds in my experiments in 12-31
        if answer == 'background':
            KB_label = [0]
            annotation_label_idx = 0
        elif obj_pair_KB_key not in self.KB:
            KB_label = [0]
            annotation_label_idx = self.ans2label[answer]
        else:
            KB_label = self.KB[obj_pair_KB_key] # list of rel_idx, which is index in logits_list
            annotation_label_idx = self.ans2label[answer]

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "decoder_prompt": decoder_prompt,
            "ref_dict": ref_dict,
            "conf": conf,
            "KB_label": KB_label,
            "annotation_label_idx": annotation_label_idx,
            "answer": answer
        }
        if self.constraint_trie is not None:
            constraint_mask = torch.zeros((len(target_item), len(self.tgt_dict))).bool()
            start_idx = len(target_item) - len(tgt_item) - 1
            for i in range(len(target_item) - len(tgt_item) - 1, len(target_item)):
                constraint_prefix_token = [self.tgt_dict.bos()] + target_item[start_idx:i].tolist()
                constraint_nodes = self.constraint_trie.get_next_layer(constraint_prefix_token)
                constraint_mask[i][constraint_nodes] = True
            example["constraint_mask"] = constraint_mask
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        is_train = 'train' in self.dataset.file_path
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos, is_train=is_train, parent_task=self.parent_task_ref)
