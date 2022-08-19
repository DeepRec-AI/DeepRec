# Co-Action Network

The following is a brief directory structure and description for this example:



```
├── data                          # Data set directory
│   ├── prepare_data.sh          # Shell script to download and process dataset
│   └── README.md               # Documentation describing how to prepare dataset
│   └── script                      # Directory contains scripts to process dataset
│       ├── data_iterator.py           
│       ├── generate_voc.py         
│       ├── local_aggretor.py               
│       ├── shuffle.py           
│       └── split_by_user.py
├── script                       #  Directory contains scripts to CAN model
│	├── Dice.py
│	├── model.py
│	├── model_avazu.py
│	├── rnn.py
│	└── utils.py
├── README.md                     # Documentation
└── train.py                      # Training script
```



## Content

[TOC]



## Model Structure

Implementation of paper "CAN: Revisiting Feature Co-Action for Click Through Rate Prediction".

paper: [arxiv (to be released)]()



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
CUDA_VISIBLE_DEVICES=0 python script/train.py train {model}

model: CAN,Cartesion,PNN, etc. (check the train.py)
```

​	



## Dataset

Amazon, Taobao and Avazu dataset is used as benchmark dataset.

### Prepare

For details of Data download, see `./data`



