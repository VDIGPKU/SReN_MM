#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE=5
    EPOCH=7
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE=5
    EPOCH=7
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
    TEST_IMDB="coco_2014_minival"
    SSTEPSIZE=5
    EPOCH=7
    ;;
  icdar)
    TRAIN_IMDB="icdar_train"
    TEST_IMDB="icdar_train"
    STEPSIZE=10
    EPOCH=20
    ;;
	icdar_mlt)
    TRAIN_IMDB="icdar_train_mlt"
    TEST_IMDB="icdar_test_mlt"
    STEPSIZE=8
    EPOCH=12
    ;;
  frame)
    TRAIN_IMDB="frame_train"
    TEST_IMDB="frame_test"
    STEPSIZE=4
    EPOCH=7
	  ;;	
  icdar_17)
    TRAIN_IMDB="icdar_train_17"
    TEST_IMDB="icdar_valid_17"
    STEPSIZE=8
    EPOCH=12
    ;;
  coco_text)
    TRAIN_IMDB="coco_text_train"
    TEST_IMDB="coco_text_val"
    STEPSIZE=8
    EPOCH=12
		;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/faster_rcnn_epoch_${EPOCH}.pth
else
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/faster_rcnn_epoch_${EPOCH}.pth
fi
set -x


if [ ! -f ${NET_FINAL}.index ]; then
  if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
      --dataset ${DATASET} \
      --tag ${EXTRA_ARGS_SLUG} \
      --epochs ${EPOCH} \
			--imdb ${TRAIN_IMDB} \
		  --imdbval ${TEST_IMDB} \
      --lr_decay_step ${STEPSIZE} \
      --net ${NET} \
      --mGPUs \
      --cuda
  else
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
      --dataset ${DATASET} \
      --epochs ${EPOCH} \
			--imdb ${TRAIN_IMDB} \
		  --imdbval ${TEST_IMDB} \
      --lr_decay_step ${STEPSIZE} \
      --net ${NET} \
      --mGPUs \
      --cuda 
  fi
fi

./experiments/scripts/test_faster_rcnn.sh $@
