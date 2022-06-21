import time
import re
import argparse
import os
import yaml


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir',
                        help='Full path of log directory',
                        required=False,
                        default='./')
    return parser


def read_config():
    bs_dic = {}
    cur_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_path, "config.yaml")
    models=[]
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f.read())
        models  = config["test_model"]
        stock_tf = config["stocktf"]
        for model in models:
            bs_dic[model]=config['model_batchsize'][model]
            
        print("=" * 30)
        print('%-20s%s'%("Model", "batch_size"))
        for model in models:
            print('%-20s%s'%(model, bs_dic[model]))
        print("=" * 30)
    return stock_tf, bs_dic, models


if __name__ == "__main__":
    stock_tf, bs_dic, models = read_config()
    parser = get_arg_parser()
    args = parser.parse_args()
    log_dir = args.log_dir

    log_list = []
    result={}
    for root, dirs, files in os.walk(log_dir, topdown=False):
        for name in files:
            if os.path.splitext(name)[1] == '.log':
                log_list.append(os.path.join(root, name))
    acc_dic = {}
    auc_dic = {}
    gstep_dic = {}
    for file in log_list:
        output = []
        file_name = os.path.split(file)[1]
        model_name = file_name.split('_')[0]
        file_name_nosurf = os.path.splitext(file_name)[0]
        with open(file, 'r') as f:
            for line in f:
                matchObj = re.search(r'global_step/sec: \d+(\.\d+)?', line)
                if matchObj:
                    output.append(matchObj.group()[17:])
                if "ACC" in line:
                    value = float(line.split()[2])
                    acc_dic[file_name_nosurf] = value
                if "AUC" in line:
                    value = float(line.split()[2])
                    auc_dic[file_name_nosurf] = value
    
        gstep = [float(i) for i in output[20:30]]
        avg = sum(gstep) / len(gstep)
        gstep_dic[file_name_nosurf] = avg

    total_dic = {}
    for model in models:
        total_dic[model]= {}
        total_dic[model]["acc"]={}
        total_dic[model]["auc"]={}
        total_dic[model]["gstep"]={}
        for acc_key in acc_dic.keys():
            if model.lower() in acc_key:
                if "tf_fp32" in acc_key:
                    total_dic[model]["acc"]["tf_fp32"]=acc_dic[acc_key]
                elif "deeprec_fp32" in acc_key:
                    total_dic[model]["acc"]["deeprec_fp32"]=acc_dic[acc_key]
                elif "deeprec_bf16" in acc_key:
                    total_dic[model]["acc"]["deeprec_bf16"]=acc_dic[acc_key]
        for auc_key in auc_dic.keys():
            if model.lower() in auc_key:
                if "tf_fp32" in auc_key:
                    total_dic[model]["auc"]["tf_fp32"]=auc_dic[auc_key]
                elif "deeprec_fp32" in auc_key:
                    total_dic[model]["auc"]["deeprec_fp32"]=auc_dic[auc_key]
                elif "deeprec_bf16" in auc_key:
                    total_dic[model]["auc"]["deeprec_bf16"]=auc_dic[auc_key]
        for gstep_key in gstep_dic.keys():
            if model.lower() in gstep_key:
                if "tf_fp32" in gstep_key:
                    total_dic[model]["gstep"]["tf_fp32"]=gstep_dic[gstep_key]
                elif "deeprec_fp32" in gstep_key:
                    total_dic[model]["gstep"]["deeprec_fp32"]=gstep_dic[gstep_key]
                elif "deeprec_bf16" in gstep_key:
                    total_dic[model]["gstep"]["deeprec_bf16"]=gstep_dic[gstep_key]            


    upgrade_dic = {}
    for model in models:
        upgrade_dic[model] = {}
        upgrade_dic[model]['tf_fp32'] = 'baseline'
        if stock_tf:
            upgrade_dic[model]['deeprec_fp32'] = total_dic[model]['gstep']['deeprec_fp32'] / total_dic[model]['gstep']['tf_fp32'] 
            upgrade_dic[model]['deeprec_bf16'] = total_dic[model]['gstep']['deeprec_bf16'] / total_dic[model]['gstep']['tf_fp32'] 

    if stock_tf:
        print("%-5s\t %10s\t %-10s\t %-10s\t %-11s\t %10s\t %10s\t %11s" %('Model', 'FrameWork', 'Datatype', 'ACC', 'AUC', 'Gstep', 'Throughput', 'Speedup'))    
        for model in total_dic.keys():
            print(model+':')
            print("%-5s\t %10s\t %-10s\t %-10.6f\t %-5.6f\t %10.2f\t %10.2f\t %11s" %('', 'StockTF', 'FP32',  total_dic[model]['acc']['tf_fp32'], total_dic[model]['auc']['tf_fp32'], total_dic[model]['gstep']['tf_fp32'], total_dic[model]['gstep']['tf_fp32']*bs_dic[model], upgrade_dic[model]['tf_fp32']))
            print("%-5s\t %10s\t %-10s\t %-10.6f\t %-5.6f\t %10.2f\t %10.2f\t %10.2f%%" %('', 'DeepRec', 'FP32',  total_dic[model]['acc']['deeprec_fp32'], total_dic[model]['auc']['deeprec_fp32'], total_dic[model]['gstep']['deeprec_fp32'], total_dic[model]['gstep']['deeprec_fp32']*bs_dic[model], upgrade_dic[model]['deeprec_fp32']*100))
            print("%-5s\t %10s\t %-10s\t %-10.6f\t %-5.6f\t %10.2f\t %10.2f\t %10.2f%%" %('', 'DeepRec', 'BF16',  total_dic[model]['acc']['deeprec_bf16'], total_dic[model]['auc']['deeprec_bf16'], total_dic[model]['gstep']['deeprec_bf16'], total_dic[model]['gstep']['deeprec_bf16']*bs_dic[model], upgrade_dic[model]['deeprec_bf16']*100))


    else:
        print("%-5s\t %10s\t %-10s\t %-10s\t %-11s\t %10s\t %10s\t" %('Model', 'FrameWork', 'Datatype', 'ACC', 'AUC', 'Gstep', 'Throughput'))
        for model in total_dic.keys():
            print(model+':')
            print("%-5s\t %10s\t %-10s\t %-10.6f\t %-5.6f\t %10.2f\t %10.2f" %('', 'DeepRec', 'FP32',  total_dic[model]['acc']['deeprec_fp32'], total_dic[model]['auc']['deeprec_fp32'], total_dic[model]['gstep']['deeprec_fp32'], total_dic[model]['gstep']['deeprec_fp32']*bs_dic[model]))
            print("%-5s\t %10s\t %-10s\t %-10.6f\t %-5.6f\t %10.2f\t %10.2f" %('', 'DeepRec', 'BF16',  total_dic[model]['acc']['deeprec_bf16'], total_dic[model]['auc']['deeprec_bf16'], total_dic[model]['gstep']['deeprec_bf16'], total_dic[model]['gstep']['deeprec_bf16']*bs_dic[model]))

   
