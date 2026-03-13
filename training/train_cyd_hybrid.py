# training/train_cyd_hybrid.py
import os
import sys

# 将项目根目录加入环境变量
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from training.trainer import do_train
from misc.utils import TrainingParams

def train_cyd_hybrid():
    class Args:
        def __init__(self):
            # 训练参数复用现有的
            self.config = '../config/config_cyd_cross.txt'
            # 指向我们第一步新写的 Hybrid 双流配置文件
            self.model_config = '../models/minkloc_hybrid.txt'
            self.debug = False

    args = Args()

    if not os.path.exists(args.config):
        print(f"错误: 找不到训练配置文件: {os.path.abspath(args.config)}")
        return
    if not os.path.exists(args.model_config):
        print(f"错误: 找不到模型配置文件: {os.path.abspath(args.model_config)}")
        print("请确保已按第一步的要求创建了 models/minkloc_hybrid.txt")
        return

    print('=' * 60)
    print('🚀 启动 CYD 混合双流 (BEV Coarse + Cross-Section Fine) 训练')
    print('=' * 60)
    print(f'Training Config : {args.config}')
    print(f'Model Config    : {args.model_config}\n')

    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.print()

    os.chdir(project_root)

    model, model_pathname = do_train(params, skip_final_eval=True)

    print('\n' + '=' * 60)
    print(f'🎉 训练结束！')
    print(f'最终模型权重已保存在: {model_pathname}_final.pth')
    print('=' * 60)

if __name__ == '__main__':
    train_cyd_hybrid()