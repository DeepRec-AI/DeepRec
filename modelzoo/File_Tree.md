# File Structure

```
modelzoo/
├── README.md
└── Model_name
    ├── README.md 
    ├── Dockerfile 
    ├── result
    │   └── README.md
    ├── data
    │   └── README.md  
    ├── distribute_k8s
    │   └── distribute_k8s.yaml  
    └── train.py
```
- README.md: A discription to model, refer to README_Template.md
- DockerFile: To build docker used in benchmark. If there will be two or more files, a DockerFile folder should create & add a README to describe these file.
- result: Default folder to save checkpoint & timeline file.
  - README.md: "Checkpoint & timeline file are default saved in this folder."
- data: Default folder to put data
  - README.md: Where to download data
- distribute_k8s：Put files related to distributed training in k8s. For example, ***distribute_k8s.yaml*** is a template yaml file to create a TFjob for distributed training.
- train_stand.py: Model training script. If there are some dependent files, create a ***"model"*** folder to put these code & add a README to describe these files.