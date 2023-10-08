# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import os
import errno
import json
import os.path as op

from tqdm import tqdm
from tsv_file import TSVFile


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_linelist_file(linelist_file):
    if linelist_file is not None:
        line_list = []
        with open(linelist_file, 'r') as fp:
            for i in fp:
                line_list.append(int(i.strip()))
        return line_list


def tsv_writer(values, tsv_file, sep='\t'):
    mkdir(op.dirname(tsv_file))
    lineidx_file = op.splitext(tsv_file)[0] + '.lineidx'
    idx = 0
    tsv_file_tmp = tsv_file + '.tmp'
    lineidx_file_tmp = lineidx_file + '.tmp'
    with open(tsv_file_tmp, 'w') as fp, open(lineidx_file_tmp, 'w') as fpidx:
        assert values is not None
        for value in values:
            assert value is not None
            # this step makes sure python2 and python3 encoded img string are the same.
            # for python2 encoded image string, it is a str class starts with "/".
            # for python3 encoded image string, it is a bytes class starts with "b'/".
            # v.decode('utf-8') converts bytes to str so the content is the same.
            # v.decode('utf-8') should only be applied to bytes class type.
            value = [v if type(v) != bytes else v.decode('utf-8') for v in value]
            v = '{0}\n'.format(sep.join(map(str, value)))
            fp.write(v)
            fpidx.write(str(idx) + '\n')
            idx = idx + len(v)
    os.rename(tsv_file_tmp, tsv_file)
    os.rename(lineidx_file_tmp, lineidx_file)


def tsv_reader(tsv_file, sep='\t'):
    with open(tsv_file, 'r') as fp:
        for i, line in enumerate(fp):
            yield [x.strip() for x in line.split(sep)]


def config_save_file(tsv_file, save_file=None, append_str='.new.tsv'):
    if save_file is not None:
        return save_file
    return op.splitext(tsv_file)[0] + append_str


def get_line_list(linelist_file=None, num_rows=None):
    if linelist_file is not None:
        return load_linelist_file(linelist_file)

    if num_rows is not None:
        return [i for i in range(num_rows)]


def generate_labelmap_file(label_file, save_file=None):
    rows = tsv_reader(label_file)
    labelmap = []
    for i, row in enumerate(rows):
        labelmap.extend(set([rect['class'] for rect in json.loads(row[1])]))
    labelmap = sorted(list(set(labelmap)))

    save_file = config_save_file(label_file, save_file, '.labelmap.tsv')
    with open(save_file, 'w') as f:
        f.write('\n'.join(labelmap))


def extract_column(tsv_file, col=1, save_file=None):
    rows = tsv_reader(tsv_file)

    def gen_rows():
        for i, row in enumerate(rows):
            row1 = [row[0], row[col]]
            yield row1

    save_file = config_save_file(tsv_file, save_file, '.col.{}.tsv'.format(col))
    tsv_writer(gen_rows(), save_file)


def remove_column(tsv_file, col=1, save_file=None):
    rows = tsv_reader(tsv_file)

    def gen_rows():
        for i, row in enumerate(rows):
            del row[col]
            yield row

    save_file = config_save_file(tsv_file, save_file, '.remove.{}.tsv'.format(col))
    tsv_writer(gen_rows(), save_file)


def generate_linelist_file(label_file, save_file=None, ignore_attrs=()):
    # generate a list of image that has labels
    # images with only ignore labels are not selected.
    line_list = []
    rows = tsv_reader(label_file)
    for i, row in tqdm(enumerate(rows)):
        labels = json.loads(row[1])
        if labels:
            if ignore_attrs and all([any([lab[attr] for attr in ignore_attrs if attr in lab])
                                     for lab in labels]):
                continue
            line_list.append([i])

    save_file = config_save_file(label_file, save_file, '.linelist.tsv')
    tsv_writer(line_list, save_file)
