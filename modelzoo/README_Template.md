# MODEL

The following is a brief directory structure and description for this example:
```
├── data                          # Data set directory
│   └── README.md                   # Documentation describing how to prepare dataset
├── distribute_k8s                # Distributed training related files
│   ├── distribute_k8s_BF16.yaml    # k8s yaml to crate a training job with BF16 feature
│   ├── distribute_k8s_FP32.yaml    # k8s yaml to crate a training job
│   └── launch.py                   # Script to set env for distributed training
├── README.md                     # Documentation
├── result                        # Output directory
│   └── README.md                   # Documentation describing output directory
└── train.py                      # Training script
```

## Content
- [MODEL](#model)
  - [Content](#content)
  - [Model Structure](#model-structure)
  - [Usage](#usage)
    - [Stand-alone Training](#stand-alone-training)
    - [Distribute Training](#distribute-training)
  - [Benchmark](#benchmark)
    - [Stand-alone Training](#stand-alone-training-1)
      - [Test Environment](#test-environment)
      - [Performance Result](#performance-result)
    - [Distribute Training](#distribute-training-1)
      - [Test Environment](#test-environment-1)
      - [Performance Result](#performance-result-1)
  - [Dataset](#dataset)
    - [Prepare](#prepare)
    - [Fields](#fields)
    - [Processing](#processing)
  - [TODO LIST](#todo-list)

## Model Structure
What is this model  
The discription of model in this folder(such as params, model size)  
The structure of the model  

## Usage
How to use MODEL model example
The first step in following chapter should be data preparation  
Put a link to #dataset, Like [Data Prepare](#prepare)

### Stand-alone Training
How to train stand-alone model

### Distribute Training
How to train distribute model


## Benchmark
The benchmark of example

### Stand-alone Training
The benchmark of stand-alone training 

#### Test Environment
The information of hardware & software stand-alone test environment

#### Performance Result  
<table>
    <tr>
        <td colspan="1"></td>
        <td>Framework</td>
        <td>DType</td>
        <td>Accuracy</td>
        <td>AUC</td>
        <td>Globalsetp/Sec</td>
    </tr>
    <tr>
        <td rowspan="3">Model Type</td>
        <td>Community TensorFlow</td>
        <td>FP32</td>
        <td></td>
        <td></td>
        <td> (baseline)</td>
    </tr>
    <tr>
        <td>DeepRec w/ oneDNN</td>
        <td>FP32</td>
        <td></td>
        <td></td>
        <td> (+1.00x)</td>
    </tr>
    <tr>
        <td>DeepRec w/ oneDNN</td>
        <td>FP32+BF16</td>
        <td></td>
        <td></td>
        <td> (+1.00x)</td>
    </tr>
</table>

- Community TensorFlow version is v1.15.5.

### Distribute Training 
The benchmark of distribute training 

#### Test Environment
The information of hardware & software stand-alone test environment


#### Performance Result  

<table>
    <tr>
        <td colspan="1"></td>
        <td>Framework</td>
        <td>Protocol</td>
        <td>DType</td>
        <td>Globalsetp/Sec</td>
    </tr>
    <tr>
        <td rowspan="3">Model Type</td>
        <td>Community TensorFlow</td>
        <td>GRPC</td>
        <td>FP32</td>
        <td></td>
    </tr>
    <tr>
        <td>DeepRec w/ oneDNN</td>
        <td>GRPC</td>
        <td>FP32</td>
        <td></td>
    </tr>
    <tr>
        <td>DeepRec w/ oneDNN</td>
        <td>GRPC</td>
        <td>FP32+BF16</td>
        <td></td>
    </tr>
</table>

- Community TensorFlow version is v1.15.5.

## Dataset
Whic dataset is used
### Prepare
Where to download dataset  
Where to put dataset  
Here put a link to data/README.md

### Fields
A detailed description of dataset

### Processing
How data are processed in this example

## TODO LIST
Next To do
- Pending to do sth