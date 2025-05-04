import torch
from collections import OrderedDict
import os
import shutil

def copy_directory_except_file(source_dir, target_dir, exclude_file='pytorch_model.bin'):
    
    os.makedirs(target_dir, exist_ok=True)
    
    for item in os.listdir(source_dir):
        source_path = os.path.join(source_dir, item)
        target_path = os.path.join(target_dir, item)
        
        if item == exclude_file:
            continue
        
        if os.path.isfile(source_path):
            shutil.copy2(source_path, target_path)

source_dir = 'pretrained_ckpt/InstructBioMol-instruct'
target_dir = f'{source_dir}-extract'
copy_directory_except_file(source_dir, target_dir)

original_model = torch.load(f'{source_dir}/pytorch_model.bin')


new_state_dict = OrderedDict()
prefix = "llama_model."

for name, param in original_model.items():
    if name.startswith(prefix):
        # remove "llama_model." prefix
        new_name = name[len(prefix):]
        new_state_dict[new_name] = param


torch.save(new_state_dict, f'{target_dir}/pytorch_model.bin')


