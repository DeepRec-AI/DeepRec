#!/bin/bash
set -e

MACHINE_TYPE=$1
IMAGE_NAME=$2
JOB_NAME=$3

ret=0
for i in $(seq 1 10); do
    [ $i -gt 1 ] && echo "WARNING: pull image failed, will retry in $((i-1)) times later" && sleep 10
    ret=0
    docker pull $IMAGE_NAME && break || ret=$?
done

if [ $ret -ne 0 ]
then
    echo "ERROR: Pull Image $IMAGE_NAME failed, exit."
    exit $ret
fi

if [ "$MACHINE_TYPE" == "gpu" ]
then
    docker run -itd --name $JOB_NAME --gpus all $IMAGE_NAME bash
else
    docker run -itd --name $JOB_NAME $IMAGE_NAME bash
fi