# DeepRec Benchmark
The following is a brief directory structure and description for this example:

```
├── benchmark_result               # Output directory  
│   ├── checkpoint                   # checkpoint file  
│   ├── log                          # Log files  
│   └── record                       # Training record  
│       ├── script                     # Run model train.py in docker  
│       └── tool                       # Benchmark auxiliary document  
│           └── check_model.sh           # Check model name file  
├── benchmark.sh                   # Benchmark script  
├── config.yaml                    # Config file  
├── log_process.py                 # Porcess log file and output reuslt  
└── README.md                      # Directory structure and how to use this script
```


## Content
- [DeepRec Benchmark](#deeprec-benchmark)
  - [Content](#content)
  - [Requirement](#requirement)
  - [Usage](#usage)
  - [Description of files](#description-of-files)
    - [config.yaml](#configyaml)
    - [log](#log)


## Requirement
The following test environment should be installed:
+ Docker : 20.10.12
+ Python : 3.6.8
+ Shyaml : 0.6.2 (installed by `pip install shyaml`)

## Usage
1. `cd` to the directory `./benchmark`  
2. modify the config files `config.yaml` to specify the parameters you want, including the test image, test model list, cpu or gpu resource, benchmark stock tf or not and parameters of deeprec benchmark
3. run the script `bash benchmark.sh`  
4. check log files in the directory `benchmark/benchmark_result/log/$CurrentTime`

## Description of files
### config.yaml
- `deeprec_test_image` : the images used to benchmark deeprec,  the default value is `alibaba/deeprec-release-modelzoo:latest`

- `tf_test_image` : the images used to benchmark stock tf, the default value is `alideeprec/deeprec-weekly-modelzoo:tf` 

- `test model` : which models you need to benchmark, like `- wide_and_deep`. These models can be added or deleted according to the benchmark needs. The default value is all models in modelzoo that need to be benchmarked

- `model_batchsize` : batch size of all models, like `wide_and_deep: 512`. Please keep the same batch size as used in model training. It is only used to calculate throughput and cannot be used to set training parameters

- `modelArgs` : parameters of deeprec benchmark, like `--emb_fusion true`, the default is `--steps 12000`. Please do not modify the `--steps` parameter

- `stocktf` : benchmark Stock tf or not, the value can be chosen in `on` and `off`. The default value is `on`, which means benchmark Stock tf. If there is no need to benchmark Stock tf, this value should be changed to `off`

- `cpu_sets` : which cpu you want to set, like `52-55,164-167`, the default is null

- `gpu_sets` : which gpu you want to set, like `all` or `device=0`, the default is null 

- `env_var` : environment config for DeepRec feature, like `export START_STATISTIC_STEP` and `export STOP_STATISTIC_STEP`

### log
The documents of each benchmark are stored in `benchmark/benchmark_result/log/$CurrentTime`

The end of log files are ACC and AUC value and the thoughtout values are multiply the average value of `global_step/sec` between 2000~12000 steps by model batchsize
