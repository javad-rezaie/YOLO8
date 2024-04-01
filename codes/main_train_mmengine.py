#
# Created on Sun Mar 31 2024
#
# Copyright (c) 2024 The Home Made AI (HOMAI)
# Author: Javad Rezaie
# License: Apache License 2.0
#


from mmengine.config import Config
from mmengine.runner import Runner
import argparse

def main(args):

    config = Config.fromfile(args.config_path)
    config.launcher = "pytorch"
    config.load_from = args.model_path
    runner = Runner.from_cfg(config)
    runner.train()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Config Path.')
    parser.add_argument('config_path', type=str, help='path to the config file')
    parser.add_argument('model_path', type=str, help='path to the model file')
    args = parser.parse_args()
    main(args)
