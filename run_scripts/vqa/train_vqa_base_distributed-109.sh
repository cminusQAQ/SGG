#!/usr/bin/env

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.
# To use the shuffled data (if exists), please uncomment the Line 24.

# Number of GPUs per GPU worker
GPUS_PER_NODE=2
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
export MASTER_ADDR=localhost
# The port for communication
export MASTER_PORT=8315
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0

data_dir=/data2/private/yutianyu/datasets/OFA_data/sgg
data=${data_dir}/20_way/train_1000lines.tsv,${data_dir}/20_way/20_way_val_200.tsv
#data=${data_dir}/20_way/20_way_train_1000line.tsv,${data_dir}/20_way/20_way_test_3025.tsv # test
#data=${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch0.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch1.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch2.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch3.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch4.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch5.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch6.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch7.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch8.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch9.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch10.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch11.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch12.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch13.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch14.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch15.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch16.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch17.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch18.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch19.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch20.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch21.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch22.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch23.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch24.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch25.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch26.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch27.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch28.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch29.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch30.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch31.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch32.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch33.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch34.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch35.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch36.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch37.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch38.tsv,${data_dir}/20_way/20_way_train_NA_ratio_4_shuffle_epoch39.tsv,${data_dir}/20_way/20_way_val_200.tsv

# Note: If you have shuffled the data in advance, please uncomment the line below.
# data=${data_dir}/vqa_train_1.tsv,${data_dir}/vqa_train_2.tsv,${data_dir}/vqa_train_3.tsv,${data_dir}/vqa_train_4.tsv,${data_dir}/vqa_train_5.tsv,${data_dir}/vqa_train_6.tsv,${data_dir}/vqa_train_7.tsv,${data_dir}/vqa_train_8.tsv,${data_dir}/vqa_train_9.tsv,${data_dir}/vqa_train_10.tsv,${data_dir}/vqa_val.tsv

ans2label_file=${data_dir}/20_way/20_way_ans2label.pkl

restore_file=/data/yutianyu/datasets/OFA_data/checkpoints/ofa_base.pt
#restore_file=/data_local/yutianyu/OFA/run_scripts/vqa/vqa_checkpoints_NA4/40_0.04_5e-5_480/checkpoint15.pt

selected_cols=0,5,2,3,4

#validate_interval=3
#tensorboard_logdir=./vqa_tensorboard_NA4_shuffle_bsz20
#log_dir=./vqa_logs_NA4_shuffle_bsz_20
#save_dir=./vqa_checkpoints_NA4_shuffle_bsz_20

validate_interval=2
tensorboard_logdir=./vqa_tensorboard_test
log_dir=./vqa_logs_test
save_dir=./vqa_checkpoints_test

mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
user_dir=../../ofa_module

task=vqa_gen
arch=ofa_base
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
batch_size=2
update_freq=4
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=128
max_object_length=30
max_tgt_length=30
num_bins=1000
patch_image_size=480

uses_ema="--uses-ema"
store_ema="--store-ema"
ema_fp32="--ema-fp32"
ema_decay=0.9999
ema_start_update=0

# Specify the inference type in validation after each fine-tuning epoch
# As mentioned in the readme, you can choose from allcand or beamsearch evaluation, default to allcand
val_inference_type=allcand

for max_epoch in {20,}; do
  echo "max_epoch "${max_epoch}
  for warmup_ratio in {0.04,}; do
    echo "warmup_updates "${warmup_updates}
    for lr in {0e-5,}; do
      echo "lr "${lr}
      for patch_image_size in {480,}; do
        echo "patch_image_size "${patch_image_size}

        log_file=${log_dir}/${max_epoch}"_"${warmup_ratio}"_"${lr}"_"${patch_image_size}"_rank"${RANK}".log"
        save_path=${save_dir}/${max_epoch}"_"${warmup_ratio}"_"${lr}"_"${patch_image_size}
        mkdir -p $save_path

        python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} ../../train.py \
            ${data} \
            --selected-cols=${selected_cols} \
            --tensorboard-logdir=${tensorboard_logdir} \
            --bpe-dir=${bpe_dir} \
            --user-dir=${user_dir} \
            --restore-file=${restore_file} \
            --reset-optimizer --reset-dataloader --reset-meters \
            --save-dir=${save_path} \
            --task=${task} \
            --arch=${arch} \
            --criterion=${criterion} \
            --label-smoothing=${label_smoothing} \
            --batch-size=${batch_size} \
            --batch-size-valid=4 \
            --update-freq=${update_freq} \
            --encoder-normalize-before \
            --decoder-normalize-before \
            --share-decoder-input-output-embed \
            --share-all-embeddings \
            --layernorm-embedding \
            --patch-layernorm-embedding \
            --code-layernorm-embedding \
            --resnet-drop-path-rate=${resnet_drop_path_rate} \
            --encoder-drop-path-rate=${encoder_drop_path_rate} \
            --decoder-drop-path-rate=${decoder_drop_path_rate} \
            --dropout=${dropout} \
            --attention-dropout=${attention_dropout} \
            --weight-decay=0.01 \
            --optimizer=adam \
            --adam-betas="(0.9,0.999)" \
            --adam-eps=1e-08 \
            --clip-norm=1.0 \
            --lr-scheduler=polynomial_decay \
            --lr=${lr} \
            --max-epoch=${max_epoch} \
            --warmup-ratio=${warmup_ratio} \
            --log-format=simple \
            --log-interval=10 \
            --fixed-validation-seed=7 \
            --keep-last-epochs=15 \
            --save-interval=${validate_interval} --validate-interval=${validate_interval} \
            --best-checkpoint-metric=vqa_score --maximize-best-checkpoint-metric \
            --max-src-length=${max_src_length} \
            --max-object-length=${max_object_length} \
            --max-tgt-length=${max_tgt_length} \
            --find-unused-parameters \
            --freeze-encoder-embedding \
            --freeze-decoder-embedding \
            --ans2label-file=${ans2label_file} \
            --valid-batch-size=10 \
            --add-type-embedding \
            --scale-attn \
            --scale-fc \
            --scale-heads \
            --disable-entangle \
            --num-bins=${num_bins} \
            --patch-image-size=${patch_image_size} \
            --prompt-type=prev_output \
            --fp16 \
            --fp16-scale-window=512 \
            --add-object \
            ${uses_ema} \
            ${store_ema} \
            ${ema_fp32} \
            --ema-decay=${ema_decay} \
            --ema-start-update=${ema_start_update} \
            --val-inference-type=${val_inference_type} \
            --num-workers=0 >> ${log_file} 2>&1
      done
    done
  done
done
