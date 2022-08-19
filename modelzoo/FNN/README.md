# FNN

The following is a brief directory structure and description for this example:



```
├── data                        # Data set directory
│   └── README.md              # Documentation describing how to prepare dataset
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

Implementation of paper "Deep Learning over Multi-field Categorical Data– A Case Study on User Response  Prediction".



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

 iPinYou dataset is used as benchmark dataset.

### Prepare

For details of Data download, see [Data Preparation](https://github.com/Atomu2014/make-ipinyou-data)

### Campaigs

We use campaign 1458 as example here.

```
make-ipinyou-data/1458$ ls
featindex.txt  test.log.txt  test.txt  train.log.txt  train.txt
```

- `train.log.txt` and `test.log.txt` are the formalised string data for each row (record) in train and test. The first column is whether the user click the ad or not.
- `featindex.txt`maps the features to their indexes. For example, `8:1.1.174.* 76` means that the 8th column in `train.log.txt` with the string `1.1.174.*` maps to feature index `76`.
- `train.txt` and `test.txt` are the mapped vector data for `train.log.txt` and `test.log.txt`. The format is y:click, and x:features. Such data is in the standard form as introduced in [iPinYou Benchmarking](http://arxiv.org/abs/1407.7073).
