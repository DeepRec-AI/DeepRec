# MODEL

- [MODEL](#model)
  - [Model Structure](#model-structure)
  - [Usage](#usage)
    - [Stand-alone Training](#stand-alone-training)
    - [Distribute Training](#distribute-training)
  - [Benchmark](#benchmark)
    - [Test Environment](#test-environment)
    - [Standing-alone training](#standing-alone-training)
    - [Distribute Training](#distribute-training-1)
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
### Test Environment
The information of hardware & software test environment

### Standing-alone training 
The benchmark of Standing-alone training 

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