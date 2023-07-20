#!/bin/bash

#This folders contains estimator and MonitoredTrainingSession examples to use collective training respectively.

echo "====================================================================="
echo "Running dcnv2 with SOK"
echo "====================================================================="

cd dcnv2

COLLECTIVE_STRATEGY=sok python3 -m tensorflow.python.distribute.launch python3 train.py --data_location ./data --steps 500 --mode train

echo "====================================================================="
echo "Running deepfm with HB"
echo "====================================================================="

cd ../deepfm

COLLECTIVE_STRATEGY=hb python3 -m tensorflow.python.distribute.launch python3 train.py --data_location ./data --steps 500 --mode train

echo "Finish!"