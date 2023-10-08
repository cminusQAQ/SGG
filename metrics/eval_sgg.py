import numpy as np
from .sgg_metrics import SGRecall, SGMeanRecall
import json

def do_vg_evaluation(inputs, vocab_file):
    ind_to_predicates = load_info(vocab_file)

    mode = 'predcls'

    num_rel_category = 21
    iou_thres = 0.5

    result_str = '\n' + '=' * 100 + '\n'
    # ----------------------------------------------------------------
    result_dict = {}
    evaluator = {}
    # tradictional Recall@K
    eval_recall = SGRecall(result_dict)
    eval_recall.register_container(mode)
    evaluator['eval_recall'] = eval_recall

    # used for meanRecall@K
    eval_mean_recall = SGMeanRecall(result_dict, len(ind_to_predicates), ind_to_predicates, print_detail=True)
    eval_mean_recall.register_container(mode)
    evaluator['eval_mean_recall'] = eval_mean_recall

    # prepare all inputs
    global_container = {}
    global_container['result_dict'] = result_dict # {}
    global_container['mode'] = mode # predcls
    global_container['num_rel_category'] = num_rel_category # 21
    global_container['iou_thres'] = iou_thres # 0.5

    for input in inputs:
        evaluate_relation_of_one_image(input, global_container, evaluator)

    # calculate mean recall
    eval_mean_recall.calculate_mean_recall(mode)

    # print result
    result_str += eval_recall.generate_print_string(mode)
    result_str += eval_mean_recall.generate_print_string(mode)
    # --------------------------------------------------------------
    result_str += '=' * 100 + '\n'

    print(result_str)

    return float(np.mean(result_dict[mode + '_recall'][100]))

def do_vg_evaluation_50(inputs):
    vocab_file = '/home/zhangfan/sup_50/VG-SGG-dicts-with-attri.json'
    ind_to_predicates = load_info(vocab_file)

    mode = 'predcls'

    num_rel_category = 51
    iou_thres = 0.5

    result_str = '\n' + '=' * 100 + '\n'
    # ----------------------------------------------------------------
    result_dict = {}
    evaluator = {}
    # tradictional Recall@K
    eval_recall = SGRecall(result_dict)
    eval_recall.register_container(mode)
    evaluator['eval_recall'] = eval_recall

    # used for meanRecall@K
    eval_mean_recall = SGMeanRecall(result_dict, len(ind_to_predicates), ind_to_predicates, print_detail=True)
    eval_mean_recall.register_container(mode)
    evaluator['eval_mean_recall'] = eval_mean_recall

    # prepare all inputs
    global_container = {}
    global_container['result_dict'] = result_dict # {}
    global_container['mode'] = mode # predcls
    global_container['num_rel_category'] = num_rel_category # 21
    global_container['iou_thres'] = iou_thres # 0.5

    for input in inputs:
        evaluate_relation_of_one_image(input, global_container, evaluator)

    # calculate mean recall
    eval_mean_recall.calculate_mean_recall(mode)

    # print result
    result_str += eval_recall.generate_print_string(mode)
    result_str += eval_mean_recall.generate_print_string(mode)
    # --------------------------------------------------------------
    result_str += '=' * 100 + '\n'

    print(result_str)

    return float(np.mean(result_dict[mode + '_recall'][100]))



def evaluate_relation_of_one_image(input, global_container, evaluator):
    """
    Returns:
        pred_to_gt: Matching from predicate to GT
        pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
        pred_triplet_scores: [cls_0score, relscore, cls1_score]
    """
    # unpack all inputs
    mode = global_container['mode']

    local_container = {}
    local_container['gt_rels'] = np.array(input['gt_triple'])
    # if there is no gt relations for current image, then skip it
    if len(local_container['gt_rels']) == 0:
        return

    local_container['gt_boxes'] = np.array(input['boxes'])  # (#gt_objs, 4)
    local_container['gt_classes'] = np.array(input['labels'])  # (#gt_objs, )
    local_container['pred_rels'] = np.array(input['pred_rels'])

    # to calculate accuracy, only consider those gt pairs
    # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing"
    # for sgcls and predcls

    if mode == 'predcls':
        local_container['pred_boxes'] = local_container['gt_boxes']
        local_container['pred_classes'] = local_container['gt_classes']

    elif mode == 'sgcls':
        if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
            print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
    elif mode == 'sgdet' or mode == 'phrdet':
        pass
    else:
        raise ValueError('invalid mode')


    # Traditional Metric with Graph Constraint
    # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
    local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container, mode)
    # Mean Recall
    evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    return

def load_info(dict_file):
    """
    Loads the file containing the visual genome label meaningslab
    """
    info = json.load(open(dict_file, 'r'))
    info['predicate_to_idx']['__background__'] = 0
    predicate_to_ind = info['predicate_to_idx']
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])
    return ind_to_predicates # 按照数字排序后，classes predicates attributes的从小到大的列表

if __name__=="__main__":
    inputs = [json.load(open("sample.json")), json.load(open("sample.json"))]
    do_vg_evaluation(inputs)
    print(load_info('./VG-SGG-dicts-with-attri.json'))

