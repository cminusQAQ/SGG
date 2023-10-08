#!/usr/bin/env

# Number of GPUs per GPU worker
GPUS_PER_NODE=1
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
export MASTER_ADDR=localhost
# The port for communication
export MASTER_PORT=8315
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0

data_dir=/data/cminus-hu/ofa
split=20_way_caption_five_filtered

ans2label_file=${data_dir}/${split}/20_way_ans2label.pkl
restore_file=${data_dir}/../SGG_0909_OFA_base.pt
#restore_file=/data_local/yutianyu/OFA/run_scripts/vqa/vqa_checkpoints_NA4/40_0.04_5e-5_480/checkpoint15.pt
#ls -alht $restore_file
#exit

validate_interval=3

#validate_interval=2
#tensorboard_logdir=./vqa_tensorboard_test
#log_dir=./vqa_logs_test
#save_dir=./vqa_checkpoints_test

bpe_dir=../../utils/BPE
user_dir=../../ofa_module

task=vqa_gen
arch=ofa_base
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
batch_size=2
batch_size_valid=1
update_freq=8
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
selected_cols=0,5,2,3,4

uses_ema="--uses-ema"
store_ema="--store-ema"
ema_fp32="--ema-fp32"
ema_decay=0.9999
ema_start_update=0

# Specify the inference type in validation after each fine-tuning epoch
# As mentioned in the readme, you can choose from allcand or beamsearch evaluation, default to allcand
val_inference_type=allcand

for NA in {1,}; do
  echo "NA "${NA}
  data=${data_dir}/${split}/query_train_1000_lines.tsv,${data_dir}/20_way_caption_five_filtered/query_val_200.tsv
#  data=${data_dir}/${split}/train_NA${NA}_E0.tsv,${data_dir}/${split}/train_NA${NA}_E1.tsv,${data_dir}/${split}/train_NA${NA}_E2.tsv,${data_dir}/${split}/train_NA${NA}_E3.tsv,${data_dir}/${split}/train_NA${NA}_E4.tsv,${data_dir}/${split}/train_NA${NA}_E5.tsv,${data_dir}/${split}/train_NA${NA}_E6.tsv,${data_dir}/${split}/train_NA${NA}_E7.tsv,${data_dir}/${split}/train_NA${NA}_E8.tsv,${data_dir}/${split}/train_NA${NA}_E9.tsv,${data_dir}/${split}/train_NA${NA}_E10.tsv,${data_dir}/${split}/train_NA${NA}_E11.tsv,${data_dir}/${split}/train_NA${NA}_E12.tsv,${data_dir}/${split}/train_NA${NA}_E13.tsv,${data_dir}/${split}/train_NA${NA}_E14.tsv,${data_dir}/${split}/train_NA${NA}_E15.tsv,${data_dir}/${split}/train_NA${NA}_E16.tsv,${data_dir}/${split}/train_NA${NA}_E17.tsv,${data_dir}/${split}/train_NA${NA}_E18.tsv,${data_dir}/${split}/train_NA${NA}_E19.tsv,${data_dir}/${split}/train_NA${NA}_E20.tsv,${data_dir}/${split}/train_NA${NA}_E21.tsv,${data_dir}/${split}/train_NA${NA}_E22.tsv,${data_dir}/${split}/train_NA${NA}_E23.tsv,${data_dir}/${split}/train_NA${NA}_E24.tsv,${data_dir}/${split}/train_NA${NA}_E25.tsv,${data_dir}/${split}/train_NA${NA}_E26.tsv,${data_dir}/${split}/train_NA${NA}_E27.tsv,${data_dir}/${split}/train_NA${NA}_E28.tsv,${data_dir}/${split}/train_NA${NA}_E29.tsv,${data_dir}/20_way/val_200.tsv
  for max_epoch in {30,}; do
    echo "max_epoch "${max_epoch}
    for warmup_ratio in {0.04,}; do
      echo "warmup_updates "${warmup_updates}
      for lr in {2e-5,}; do
        echo "lr "${lr}
        for patch_image_size in {480,}; do
          echo "patch_image_size "${patch_image_size}

          exp_tag=test_NA_ratio
          tensorboard_logdir=./vqa_tensorboard/${exp_tag}
          log_dir=./vqa_logs/${exp_tag}
          save_dir=./vqa_checkpoints/${exp_tag}
          mkdir -p $log_dir $save_dir $tensorboard_logdir

          log_file=${log_dir}/${NA}"_B"${batch_size}"_A"${update_freq}"_E"${max_epoch}"_"${warmup_ratio}"_"${lr}"_"${patch_image_size}"_rank"${RANK}".log"
          save_path=${save_dir}/${NA}"_B"${batch_size}"_A"${update_freq}"_E"${max_epoch}"_"${warmup_ratio}"_"${lr}"_"${patch_image_size}
          mkdir -p $save_path
          echo "qwq"
          python3  ../../train.py \
              ${data} \
              --selected-cols=${selected_cols} \
              --data-buffer-size 10 \
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
              --label-proxy ${label_proxy} \
              --distill default \
              --distill-alpha=${distill_alpha} \
              --batch-size=${batch_size} \
              --batch-size-valid=${batch_size_valid} \
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
              --save-interval=${validate_interval} --validate-interval=${validate_interval} \
            --save-interval-updates=${validate_interval_updates} --validate-interval-updates=${validate_interval_updates} \
              --best-checkpoint-metric=R@100 --maximize-best-checkpoint-metric \
              --max-src-length=${max_src_length} \
              --max-object-length=${max_object_length} \
              --max-tgt-length=${max_tgt_length} \
              --find-unused-parameters \
              --freeze-encoder-embedding \
              --freeze-decoder-embedding \
              --ans2label-file=${ans2label_file} \
              --valid-batch-size=51 \
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
              --val-inference-type=${val_inference_type}
        done
      done
    done
  done
done
# b, w, reserve
# 2, 3, 5        7h   0G
# 1, 5, 5        7h -10G
