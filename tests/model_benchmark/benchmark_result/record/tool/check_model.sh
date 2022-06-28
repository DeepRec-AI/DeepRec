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

# modify the shell command line to comments, by given column id or regexp
function modify_text()
{
        regex1=$1
        regex2=$2
        model_name=$3   
        col_id=($(cat $src_file | grep -n $regex1 | grep $regex2  |awk -F ':' '{print $1}'))
        count=${#col_id[@]}
        for i in $(seq 1 $count)
        do
                sed -i "${col_id[i-1]}c # [Error] No model named ${model_name}, related command line has been deleted..." $src_file
        done
}


currentTime=$1
test_list=$2

exist_list=$(ls -F /root/modelzoo | grep '/'|sed 's/\///g')
src_file=/benchmark_result/record/script/$currentTime/benchmark_modelzoo.sh
log_file="/benchmark_result/log/$currentTime/error.log"


for model in $test_list
do
        if [[ -z $(echo $exist_list | grep $model) ]];then
                
                echo "[ERROR] Model $model does not exists, related command will be skipped" >> $log_file
                modify_text  "echo" $model $model
                modify_text  "cd" "/root/modelzoo/$model" $model
                modify_text  "train.py" "${model,,}" $model
        fi
done

[[ -f $log_file ]] && echoColor red "$(cat  $log_file)"