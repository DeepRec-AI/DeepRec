#!/bin/bash
function echoColor() {
	case $1 in
	green)
		echo -e "\033[32;49m$2\033[0m"
		;;
	red)
		echo -e "\033[31;40m$2\033[0m"
		;;
	*)
		echo "Example: echo_color red string"
		;;
	esac
}

function echoBanner()
{
    echoColor green "#############################################################################################################################"
    echoColor green "#   ####:                       #####                       #####.                      #                           #       #"     
    echoColor green "#   #  :#.                      #    #                      #   :#                      #                           #       #"     
    echoColor green "#   #   :#  ###    ###   # ##   #    #  ###     ##:         #    #  ###   #:##:    ##:  #:##:  ## #   .###.   #:##: #  :#   #" 
    echoColor green "#   #    #    :#     :#  #   #  #   :#    :#   #            #   :#    :#  #  :#   #     #  :#  #:#:#  #: :#   ##  # # :#    #"  
    echoColor green "#   #    # #   #  #   #  #   #  #####  #   #  #.            #####. #   #  #   #  #.     #   #  # # #      #   #     #:#     #"   
    echoColor green "#   #    # #####  #####  #   #  #  .#: #####  #             #   :# #####  #   #  #      #   #  # # #  :####   #     ##      #"    
    echoColor green "#   #   :# #      #      #   #  #   .# #      #.            #    # #      #   #  #.     #   #  # # #  #:  #   #     #.#.    #"  
    echoColor green "#   #  :#.     #      #  #   #  #    #     #   #            #   :#     #  #   #   #     #   #  # # #  #.  #   #     # .#    #" 
    echoColor green "#   ####:   ###:   ###:  # ##   #    :  ###:    ##:         #####.  ###:  #   #    ##:  #   #  # # #  :##:#   #     #  :#   #" 
    echoColor green "#                        #                                                                                                  #"                                                                                               
    echoColor green "#                        #                                                                                                  #"                                                                                                
    echoColor green "#                        #                                                                                                  #"                                                                                                
    echoColor green "#############################################################################################################################"
} 

function make_script()
{
    script=$script_path

    [[ ! -d $(dirname $script) ]] && mkdir -p $(dirname $script)

    paras=$modelArgs
    
    echo "model_list=\$1" >>$script
    echo "category=\$2" >>$script
    echo "cat_param=\$3" >>$script
    echo "check=\$4" >> $script

    echo "$env_var" >> $script && echo "">> $script

    echo " " >> $script && echo "[[ \$check == true ]] && bash /benchmark_result/record/tool/check_model.sh $currentTime \"\${model_list[*]}\" " >>$script

    for line in $model_list
    do
        log_tag=$(echo $paras| sed 's/--/_/g' | sed 's/ //g')
        [[ $paras == "" ]] && log_tag=""
        model_name=$line
        bs=$(cat config.yaml | shyaml get-value model_batchsize | grep $model_name | awk -F ":" '{print $2}')
        echo "echo 'Testing $model_name  $paras ...'" >> $script
        echo "cd /root/modelzoo/$model_name/" >> $script
        [[ ! -d  $checkpoint_dir/$currentTime/${model_name,,}_script$$log_tag ]]\
        &&mkdir -p $checkpoint_dir/$currentTime/${model_name,,}_$script$log_tag
        newline="LD_PRELOAD=/root/modelzoo/libjemalloc.so.2.5.1 python train.py --batch_size $bs \$cat_param $paras  --checkpoint /benchmark_result/checkpoint/$currentTime/${model_name,,}_\${category}$log_tag  >/benchmark_result/log/$currentTime/${model_name,,}_\${category}$log_tag.log 2>&1"
        echo $newline >> $script
    done
}

# check container environment
function checkEnv()
{
    status1=$( sudo docker ps -a | grep deeprec_bf16)
    status2=$( sudo docker ps -a | grep deeprec_fp32)
    status3=$( sudo docker ps -a | grep tf_fp32)
    if [[  -n $status1 ]];then
        echoColor red "[Warning] Container named deeprec_bf16 may have already existed. Please check your environment to ensure it won't have an effect on the benchmark performance"
    fi
    if [[  -n $status2 ]];then
        echoColor red "[Warning] Container named deeprec_fp32 may have already existed. Please check your environment to ensure it won't have an effect on the benchmark performance"
    fi
    if [[  -n $status3 ]];then
        echoColor red "[Warning] Container named tf_fp32 may have already existed. Please check your environment to ensure it won't have an effect on the benchmark performance"
    fi
}

# run containers
function runContainers()
{
    sudo docker pull $deeprec_test_image
    echoColor green "[INFO] Testing Deeprec BF16..."
    runSingleContainer $deeprec_test_image deeprec_bf16 true
    echoColor green "[INFO] Testing Deeprec FP32..."
    runSingleContainer $deeprec_test_image deeprec_fp32

    if [[ $stocktf==True ]];then
        echoColor green "[INFO] Testing Stock TF FP32..."
        sudo docker pull $tf_test_image
        runSingleContainer $tf_test_image tf_fp32
    fi
}

function runSingleContainer()
{
    image_repo=$1
    cat_name=$2
    check=$3
    script_name='benchmark_modelzoo.sh'

    if [[ $cat_name == "tf_fp32" ]];then
        param="--tf"
    elif [[ $cat_name == "deeprec_bf16" ]];then
        param="--bf16"
    else
        param=
    fi

    container_name=$(echo $2 | awk -F "." '{print $1}')
    host_path=$(cd benchmark_result && pwd)
    sudo docker run -it $cpu_optional $gpu_optional \
                --rm \
                --name $container_name-$short_time \
                -v $host_path:/benchmark_result/\
                $image_repo /bin/bash /benchmark_result/record/script/$currentTime/$script_name "${model_list[*]}" $cat_name $param  $check
}


function main()
{
    echoBanner
    make_script\
    && checkEnv\
    && runContainers\
    && python3 ./log_process.py --log_dir=$log_dir/$currentTime
}

# time
currentTime=`date "+%Y-%m-%d-%H-%M-%S"`
short_time=$(echo $currentTime | cut -c 6-19)

# config_file
config_file="./config.yaml"

# Args
modelArgs=$(cat $config_file | shyaml get-value modelArgs)


# directory
log_dir=$(cd ./benchmark_result/log/ && pwd)
checkpoint_dir=$(cd ./benchmark_result/checkpoint/ && pwd)

# run.sh
script_path="./benchmark_result/record/script/$currentTime/benchmark_modelzoo.sh"

# model list
model_list=$(cat config.yaml | shyaml get-values test_model)

# stocktf
stocktf=$(cat $config_file | shyaml get-value stocktf)

# cpus
cpus=$(cat $config_file | shyaml get-value cpu_sets)

# gpus
gpus=$(cat $config_file | shyaml get-value gpu_sets)

# image name
deeprec_test_image=$(cat $config_file | shyaml get-value deeprec_test_image)
tf_test_image=$(cat $config_file | shyaml get-value tf_test_image)

# environment variables
env_var=$(cat $config_file | shyaml get-values env_var)

# check config file
if [[ ! -f $config_file ]];then
    echoColor red "[Error] The config file does not exists"
    exit -1
fi

[[ $modelArgs == None ]] && modelArgs=
[[ $cpus != 'None' ]] && cpu_optional="--cpuset-cpus $cpus"
[[ $gpus != 'None' ]] && gpu_optional="--gpus $gpus"
[ ! -d $log_dir/$currentTime ] && mkdir -p "$log_dir/$currentTime"
[ ! -d $checkpoint_dir/$currentTime ] && mkdir -p "$checkpoint_dir/$currentTime"

main