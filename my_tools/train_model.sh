#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./my_tools/train_model.sh 0 VGG16 sign data/imagenet_models/VGG16.v2.caffemodel  faster_rcnn_end2end \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"
# Example : Resnet
# ./my_tools/train_model.sh 0 ResNet101_BN_SCALE_Merged_OHEM faster_rcnn_end2end sign data/imagenet_models/ResNet101_BN_SCALE_Merged.caffemodel   

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
METHOD=$3
DATASET=$4
INIT_MODEL=$5

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in 
  sign)
    #my own sign model
    TRAIN_IMDB="sign_2017_train"
    TEST_IMDB="sign_2017_trainval"
    PT_DIR="sign"
    ITERS=490000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/faster_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/${METHOD}/solver.prototxt \
  --weights ${INIT_MODEL}  \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/${METHOD}.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x
echo final net "$NET_FINAL"
time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/${METHOD}/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/${METHOD}.yml \
#for method :ohem may be try add below?
#  --num_dets 2000 \
#  --det_thresh 0.00001 \
  ${EXTRA_ARGS}
