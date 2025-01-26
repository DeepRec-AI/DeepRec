# FNN

The following is a brief directory structure and description for this example:



```
├── data                        # Data set directory
│   ├── prepare_data.sh         # Shell script to download and process dataset
│   └── README.md              # Documentation describing how to prepare dataset
│	└──script                   # Directory contains scripts to process dataset
│       ├──data2labelencode           # Convert data to csv file
│       ├── generate_neg.py           # Create negative sample
│       ├── generate_voc.py           # Create a list of features
│       ├── history_behavior_list.py  # Count user's history behaviors
│       ├── item_map.py               # Create a map between item id and cate
│       ├── local_aggretor.py         # Generate sample data
│       ├── pick2txt.py               # Convert voc's format
│       ├── process_data.py           # Parse raw json data
│       └── split_by_user.py          # Divide the dataset
├── script                       # model set directory
│	├── contrib                  #Directory contains rnn
│	├── estimator                #Directory contains estimator to data
│	├── layers                   #Directory contains layers of model 
│	├── models                   #Directory contains FNN model
│	├── feature_column.py        # Feature marker
│	├── inputs.py                #Construction of Input Layer
│	└──utils
├── train.py                    # Training script
└── README.md                      # Documentation
```



## Content

[TOC]



## Model Structure

Implementation of paper "Deep Learning over Multi-field Categorical Data A Case Study on User Response Prediction".



## Usage

### Stand-alone Training

1. Please prepare the data set and DeepRec env.

   1. Manually

      - Follow [dataset preparation](https://github.com/alibaba/DeepRec/tree/main/modelzoo/DIEN#prepare) to prepare data set.
      - Download code by `git clone https://github.com/alibaba/DeepRec`
      - Follow [How to Build](https://github.com/alibaba/DeepRec#how-to-build) to build DeepRec whl package and install by `pip install $DEEPREC_WHL`.

   2. Docker(Recommended)

      ```
      docker pull alideeprec/deeprec-release-modelzoo:latest
      docker run -it alideeprec/deeprec-release-modelzoo:latest /bin/bash
      
      # In docker container
      cd /root/modelzoo/CAN
      ```

​	2.train.

```
  python train.py
```

​	



## Dataset

 Amazon Dataset Books dataset is used as benchmark dataset.

### Prepare

For details of Data download, see [Data Preparation](https://github.com/Atomu2014/make-ipinyou-data)
