#!/bin/bash
set -e

if [ $# -ne 4 ]
then
    echo "Usage:\n\tinstall.sh REPO_URL MACHINE_IP_FILE PASSWORD LABEL"
    exit
fi

REPO_URL=$1
MACHINE_IP_FILE=$2
PASSWORD=$3
LABEL=$4

for ip in `cat ${MACHINE_IP_FILE}`; do
    echo "====================================================================="
    echo "Dealing with machine: ${ip}"
    echo "====================================================================="
    sshpass -p ${PASSWORD} rsync -e 'ssh -o StrictHostKeyChecking=no' ./runner_setup.sh root@${ip}:/root
    read -p "Runner Token: " token
    sshpass -p ${PASSWORD} ssh root@${ip} bash runner_setup.sh ${REPO_URL} ${token} ${LABEL}
done


echo "Finish!"
