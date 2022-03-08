#!/bin/bash
set -e

JOBNAME=$1

docker inspect $JOBNAME > /dev/null || is_exist=$?

if [ $is_exist -eq 0 ]
then
    docker stop $JOB_NAME
    docker rm $JOB_NAME
fi

exit 0