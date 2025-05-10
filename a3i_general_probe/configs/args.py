import argparse
import json
from collections import (
    deque,
)  # just for fun using dequeue instead of just a list for faster appends
from pprint import pprint



class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        data_dict = json.load(values)
        arg_list = deque()
        action_dict = {e.option_strings[0]: e for e in parser._actions}
        for i, e in enumerate(data_dict):
            arg_list.extend(self.__build_parse_arge__(e, data_dict, action_dict))
        parser.parse_args(arg_list, namespace=namespace)

    def __build_parse_arge__(self, arg_key, arg_dict, file_action):
        arg_name = f"--{arg_key}"
        arg_val = str(arg_dict[arg_key]).replace(
            "'", '"'
        )  # list of text need to be modified so they can be parsed properly
        try:
            file_action[arg_name].required = False
        except:
            raise KeyError(
                f"The Key {arg_name} is not an expected parameter. Delete it from config or update build_args method in helper_utils.configs.py"
            )
        return arg_name, arg_val


def parse_bool(s: str): 
    if s.lower() == 'true': 
        return  True  
    if s.lower() == 'false': 
        return  False  
    else: 
        raise Exception("Typo in bool var please check")

def get_model_choices():
    return  ["hello"]
"""
def get_trainer_choices():
    return list(TrainerRegister.get_trainers())
def get_dataset_choices():
    return list(DatasetRegister.get_datasets())
"""

def build_train_args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--csv_path',required=True,type=str) 
    parser.add_argument('--config_path',required=False,type=open,action=LoadFromFile)
    parser.add_argument('--log_dir',required=True,type=str) 
    parser.add_argument('--device',required=True,type=json.loads) 
    parser.add_argument('--learn_rate',required=True,type=float)
    parser.add_argument("--weight_decay",required=True,type=float)
    parser.add_argument("--batch_agg",required=True,type=int)
    parser.add_argument('--epochs',required=True,type=int) 
    parser.add_argument('--train_mode',required=True,type=str)
    parser.add_argument("--dataset",required=True,type=str)
    parser.add_argument("--weight_loss",required=False,default=False,type=parse_bool)
    parser.add_argument("--model",required=True)
    parser.add_argument('--batch_size',required=False,default=1,type=int)
    parser.add_argument("--model_parameters",required=False,default=None,type=json.loads)
    parser.add_argument("--debug",required=False,default=False,type=parse_bool)
    parser.add_argument("--col_info",required=True,type=json.loads,default=None)
    parser.add_argument("--model_weight",required=False,default="",type=str)
    parser.add_argument("--trainer_args",required=False,default=None,type=json.loads)
    return parser 

def build_infer_args():
    parser = argparse.ArgumentParser(
        description="Confguration for my deep learning model training for segmentation"
    )
    parser.add_argument(
        "--config_path", required=False, type=open, action=LoadFromFile, help="Path"
    )
    parser.add_argument("--csv_path", required=True, type=str)
    ##image specifics
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument(
        "--device",
        type=json.loads,
        required=True,
        default=["cuda:0"],
        help="GPU parameter",
    )
    parser.add_argument("--model_weight", type=str, required=True)
    parser.add_argument("--seed", type=int, default=349)
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--splits", type=json.loads, default=["test"])
    parser.add_argument("--sampler", type=str, default="none", required=False)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--image_func", required=True, default="dcm", type=str)
    parser.add_argument("--col_info", required=True, type=json.loads)
    parser.add_argument("--grad_step", required=False, default=1, type=int)
    parser.add_argument("--do_adapt",required=False,default=True,type=parse_bool)
    parser.add_argument("--debug",required=False,default=False,type=parse_bool)
    parser.add_argument( "--test_transforms",
        type=json.loads,
        required=True,
        help="List of Names of test transforms and augmentations in form [load] should be subset of train transforms",
    )  # TODO: asert test is subset of train excluding rands
    parser.add_argument("--img_size", required=True, type=json.loads)
    parser.add_argument("--output_dir",required=True,type=str)
    parser.add_argument("--trainer_args",required=False,type=json.loads,default="{}")
    return parser

def build_feat_extract_args(): 
    parser = argparse.ArgumentParser(
        description="Configruaiton for extracting feature embeddings"
    )
    parser.add_argument(
        "--config_path", required=False, type=open, action=LoadFromFile, help="Path"
    )
    parser.add_argument("--csv_path", required=True, type=str)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument(
        "--device",
        type=json.loads,
        required=True,
        default=["cuda:0"],
        help="GPU parameter",
    )
    parser.add_argument("--seed", type=int, default=349)
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--image_func", required=True, default="dcm", type=str)
    parser.add_argument("--col_info", required=True, type=json.loads)
    parser.add_argument( "--test_transforms",
        type=json.loads,
        required=True,
        help="List of Names of test transforms and augmentations in form [load] should be subset of train transforms",
    )  # TODO: asert test is subset of train excluding rands 
    parser.add_argument("--transform_conf",required=True,type=json.loads)
    parser.add_argument("--output_dir",required=True,type=str)
    parser.add_argument("--extractor_args",required=False,type=json.loads,default="{}")
    parser.add_argument("--debug",required=False,type=parse_bool,default=False)
    parser.add_argument("--model",required=True,type=str)
    parser.add_argument("--model_parameters",required=True,type=json.loads)
    parser.add_argument("--extractor",required=True,type=str)
    parser.add_argument("--mapping_path",required=True,type=str)
    return parser


def get_feat_extract_args(): 
    parser = build_feat_extract_args() 
    args = parser.parse_args() 
    conf = vars(args)
    return conf  
def get_infer_args():
    parser = build_infer_args()
    args = parser.parse_args()
    conf = vars(args)
    return conf

def get_train_args():
    parser = build_train_args()
    args = parser.parse_args()
    conf = vars(args)
    return conf


def get_test_args():
    parser = build_test_args()
    args = parser.parse_args()
    conf = vars(args)
    return conf


def print_expected_keywords(parser):
    print("{")
    for action in parser._actions:
        if action.nargs == argparse.REMAINDER:
            continue
        if action.metavar is None and action.required:
            print(f'"{action.dest}":{action.type}')
    print("}")


if __name__ == "__main__":
    import sys

    input_mode = sys.argv[1]
    print(f"Configs for {input_mode}")
    if input_mode == "train":
        parser = build_train_args()
    if input_mode == "test":
        parser = build_test_args()
    print_expected_keywords(parser)
