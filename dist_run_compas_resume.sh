#!/bin/bash
if [ $# -lt 3 ]; then
    echo "USAGE: $0 DATASET METHOD CONFIG ..."
    exit
fi

WORK_DIR='work_dirs'
DATASET=$1
METHOD=$2 # box_compas
CONFIG=$3

BACKBONE='faster'

case $CONFIG in
  *"partial"*)
    TRAINING_SETTING='partial'
    ;;
  *"mixed"*)
    TRAINING_SETTING='mixed'
    ;;
esac


GPUS=${GPUS:-4}
IMGS_PER_GPU=${IMGS_PER_GPU:-2}

PORT=${PORT:-29504}
TRAIN_STEP=${TRAIN_STEP:-10}
QUERY_UNIT=${QUERY_UNIT:-'box'}
INIT_NUM=${INIT_NUM:-2000}
ADD_NUM=${ADD_NUM:-1000}
DELETE_MODEL=${DELETE_MODEL:-0}
seed=${SEED:-2022}
START_ITER=${START_ITER:-0}

SUFFIX=""
if [ ${QUERY_UNIT} == 'box' ]; then
  SUFFIX="--count-box ${SUFFIX}"
fi

export PYTHONPATH="$(dirname $0)":$PYTHONPATH
branch_name=`git branch --show-current`
short_sha=`git rev-parse --short=11 HEAD`

epochs=""
if [[ ${@:4} == *"epochs"* ]]; then
  tmpvar=${@:4}
  tmpvar=${tmpvar#*epochs}
  epochs="_e`echo ${tmpvar%--*} | sed 's/ //g'`"
fi

TEMPLATE=${DATASET}_${BACKBONE}_${QUERY_UNIT}_${INIT_NUM}_${ADD_NUM}${epochs}_template_${seed}_${branch_name}_${short_sha}
TIMESTAMP=${DATASET}_${BACKBONE}_${QUERY_UNIT}_${INIT_NUM}_${ADD_NUM}${epochs}_${TRAINING_SETTING}_${METHOD}_$(date "+%Y%m%d%H%M%S")_${seed}_${branch_name}_${short_sha}

if [ -d "${WORK_DIR}/$TEMPLATE" ]; then
  echo "Copying from $TEMPLATE to $TIMESTAMP ..."
  cp -rdT ${WORK_DIR}/${TEMPLATE} ${WORK_DIR}/${TIMESTAMP}
  train_ini=false
else
  mkdir -p ${WORK_DIR}/${TIMESTAMP}
  train_ini=true
fi

for ((i=${START_ITER};i<${TRAIN_STEP};i++))
do
  workdir=${WORK_DIR}/${TIMESTAMP}/step_${i}
  mkdir -p ${workdir}

    if [ ${i} == 0 ]
    then
        if ${train_ini}
        then
            rm ${workdir}/*.pth
            python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
                $(dirname "$0")/tools/train.py $CONFIG --initial-ratio=${INIT_NUM} \
                --cfg-options data.samples_per_gpu=${IMGS_PER_GPU} data.workers_per_gpu=${IMGS_PER_GPU} \
                --work-dir=${workdir} --ratio=0 --seed=$seed --training-setting=${TRAINING_SETTING} \
                --mining-method=${METHOD} ${SUFFIX} --launcher pytorch ${@:4}
        fi
  else
      j=`expr ${i} - 1`
      prev_workdir=${WORK_DIR}/${TIMESTAMP}/step_${j}
      ckpt="$(ls  ${prev_workdir}/*.pth | sort -V | tail -n1)"
      python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
          $(dirname "$0")/tools/train.py $CONFIG \
          --model-result=${prev_workdir}/results.compas.json \
          --cfg-options data.samples_per_gpu=${IMGS_PER_GPU} data.workers_per_gpu=${IMGS_PER_GPU} \
                        data.train.labeled.ann_file=${prev_workdir}/labeled.json \
                        data.train.mixed.ann_file=${prev_workdir}/mixed.json \
                        data.train.partial.ann_file=${prev_workdir}/partial.json \
                        data.train.unlabeled.ann_file=${prev_workdir}/unlabeled.json \
          --work-dir=${workdir} --ratio=${ADD_NUM} --seed=$seed --training-setting=${TRAINING_SETTING} \
          --load-from-prev=${ckpt} \
          --mining-method=${METHOD} ${SUFFIX} --launcher pytorch ${@:4}
  fi

  # skip the last ratio(no data in unlabled annotation file)
  if [ ${i} != `expr ${TRAIN_STEP} - 1` ]; then
      ckpt="$(ls  ${workdir}/*.pth | sort -V | tail -n1)"
      python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
          $(dirname "$0")/tools/test_compas.py $CONFIG ${ckpt} --seed=$seed --work-dir=${workdir} \
          --cfg-options data.pool.ann_file=${workdir}/mixed.json \
          --out ${workdir}/results.pkl --format-only --gpu-collect --active-cycle ${i} --launcher pytorch
  fi
done

if [ ${DELETE_MODEL} == 1 ]; then
    find ${WORK_DIR}/${TIMESTAMP} -name '*.pth' -type f -print -exec rm -rf {} \;
fi