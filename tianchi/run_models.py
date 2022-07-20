# /bin/env python
# -*- coding: utf-8 -*-

import os
import shutil

result_dir = "/tmp/tianchi/result/"

def run_models():
    models_dir = '/tianchi/models/'
    models_name = ['DIEN', 'DIN', 'DLRM', 'DeepFM', 'MMoE', 'WDL']
    model_standard_auc =  {
                  'DIEN': 0.745,
                  'DIN': 0.707,
                  'DLRM': 0,
                  'DeepFM': 0,
                  'MMoE': 0.753,
                  'WDL': 0} 
    model_steps =  {
                  'DIEN': 2700,
                  'DIN': 16000,
                  'DLRM': 11600,
                  'DeepFM': 700,
                  'MMoE': 27700,
                  'WDL': 8900}

    model_total_speed = 0
    model_scores = {}
    is_valid = True
    # 依次执行各个模型
    for model in models_name:
        model_path = models_dir + model
        os.system('cd '+model_path+'; python train.py --steps '+str(model_steps[model]))
        with open(result_dir+model+'/result', 'r') as f:
            lines = f.readlines()
        model_speed = float(lines[0])
        model_auc = float(lines[1])
        model_scores[model+' (s)'] = str(model_speed)
        model_scores[model+'_auc'] = str(model_auc)
        if model_auc < model_standard_auc[model]:
            is_valid = False
            # model_scores[model+' (s)'] = "Failed AUC: "+str(model_auc)
            model_scores[model+' (s)'] = str(0)
        model_total_speed += model_speed

    if is_valid == False:
        model_total_speed = 864000 # 10 day seconds, max value
    model_scores['score'] = str(model_total_speed)


    return model_total_speed, model_scores

def main():
    # 执行清理
    shutil.rmtree(result_dir, ignore_errors=True)
    os.makedirs(result_dir, exist_ok=True)

    # 执行测试程序
    total_speed, model_scores = run_models()

    # 输出结果
    print('Total Time: ' + str(total_speed) + ' (s)')
    print(model_scores)

if __name__ == '__main__':
    main()
