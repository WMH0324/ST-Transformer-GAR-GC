import torch
import yaml
from GARMain import GARMain
from pprint import pprint


def init_Config():
    # cfgPath = '/kaggle/input/0509gac/20230802GARFuncKaggleCAE/config/config.yaml'
    cfgPath = 'config/config.yaml'
    # cfgPath = '/kaggle/input/0509gac/20230802GARCADKaggle/config/config_volleyball.yaml'
    with open(cfgPath) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['trainOrVal'] = 'train'
    return config


def main():
    # ===========================初始化Config Begin==============================#
    cfg = init_Config()
    pprint(cfg)
    # ===========================初始化Config End==============================#

    # ===========================调用GARMain Begin==============================#
    garMain = GARMain(cfg)
    if cfg['trainOrVal'] == 'train':
        print('Start Train....................')
        garMain.train()
    elif cfg['trainOrVal'] == 'val':
        print('Start Val....................')
        garMain.val()
    else:
        print('Start Train and Val....................')
    # ===========================调用GARMain End==============================#


if __name__ == '__main__':
    main()
