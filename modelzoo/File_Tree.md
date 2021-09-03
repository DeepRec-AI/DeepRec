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
    ├── train_stand.py
    └── train_distribute.py
```
- README.md: A discription to model, refer to README_Template.md
- DockerFile: To build docker used in benchmark. If there will be two or more files, a DockerFile folder should create & add a README to describe these file.
- result: Default folder to save checkpoint & timeline file.
  - README.md: "Checkpoint & timeline file are default saved in this folder."
- data: Default folder to put data
  - README.md: Where to download data
- train_stand.py: Stand-alone training script. If there are some dependent files, create a ***"train_stand"*** folder to put these code & add a README to describe these files. If the optimized code changes a lot, put them into ***"train_stand"*** folder and name as ***"train_stand_base.py", "train_stand_optimized.py" or "train_stand_optimized_BF16.py"***
- train_distribute.py: Distribute training script. If there are some dependent files,such as ***"k8s.yaml"***, ***"model.py"***, create a ***"train_distribute"*** folder to put these code & add a README to describe these files. If the optimized code changes a lot, put them into ***"train_distribute"*** folder and name as ***"train_distribute_base.py", "train_distribute_optimized.py" or "train_distribute_optimized_BF16.py"***