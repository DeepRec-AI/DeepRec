#!/bin/bash

#use_feature_column
horovodrun -np 4 -H localhost:4 python train.py  --steps 10000  --use_feature_columns

#embedding_variable
horovodrun -np 4 -H localhost:4 python train.py  --steps 10000 