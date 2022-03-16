#!/bin/bash
set -e

JOBNAME=$1

is_exist=0
docker inspect $JOBNAME > /dev/null || is_exist=$?

if [ $is_exist -eq 0 ]
then
    docker stop $JOBNAME
    docker rm $JOBNAME
fi

exit 0