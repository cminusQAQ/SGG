#!/usr/bin/env

# Number of GPUs per GPU worker
GPUS_PER_NODE=10
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
export MASTER_ADDR=localhost
# The port for communication
export MASTER_PORT=8315
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0

data_dir=/data_local/yutianyu/datasets/OFA_data/sgg
split=20_way_combine

ans2label_file=${data_dir}/${split}/${split}_ans2label.pkl
restore_file=${data_dir}/../checkpoints/ofa_base.pt

if [ -s ${restore_file} ]
then
  echo "Use checkpoint from "${restore_file}
  ls -alht ${restore_file}
else
  ls -alht ${restore_file}
  echo "Checkpoint file "${restore_file}" not exists!"
  exit
fi

validate_interval=10
validate_interval_updates=3000
KB_repeat=8
caption_repeat=2

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
batch_size=3
batch_size_valid=1
update_freq=1
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
template=query
label_proxy=answer # answer, selected_token

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
#  data=${data_dir}/${split}/query_train_1000_lines.tsv,${data_dir}/20_way/${template}_val_500.tsv
  data=${data_dir}/${split}/10card_card-bsz${batch_size}_NA${NA}_KB${KB_repeat}_caption${caption_repeat}_loop10000.tsv,${data_dir}/${split}/${template}_val_500.tsv
  for max_epoch in {1,}; do
    echo "max_epoch "${max_epoch}
    for warmup_ratio in {0.04,}; do
      echo "warmup_updates "${warmup_updates}
      for lr in {5e-5,}; do
        echo "lr "${lr}
        for patch_image_size in {480,}; do
          echo "patch_image_size "${patch_image_size}

          exp_tag=20_way_combine_KB${KB_repeat}_caption${caption_repeat}
          tensorboard_logdir=./vqa_tensorboard/${exp_tag}
          log_dir=./vqa_logs/${exp_tag}
          save_dir=./vqa_checkpoints/${exp_tag}
          mkdir -p $log_dir $save_dir $tensorboard_logdir

          log_file=${log_dir}/${NA}"_B"${batch_size}"_A"${update_freq}"_E"${max_epoch}"_"${warmup_ratio}"_"${lr}"_"${patch_image_size}"_rank"${RANK}".log"
          save_path=${save_dir}/${NA}"_B"${batch_size}"_A"${update_freq}"_E"${max_epoch}"_"${warmup_ratio}"_"${lr}"_"${patch_image_size}
          mkdir -p $save_path

          python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} ../../train.py \
            ${data} \
            --selected-cols=${selected_cols} \
            --data-buffer-size 5 \
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
            --valid-batch-size=26 \
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
            --num-workers=5 >>${log_file} 2>&1
        done
      done
    done
  done
done
