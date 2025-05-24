## Advancing Biomolecule Understanding and Design Following Human Instructions

This repository is the official implementation of [InstructBioMol](https://arxiv.org/abs/2410.07919).

### 🔔 News
- 2025.05, InstructBioMol is accepted for publication in Nature Machine Intelligence.

### 💡 Brief Introduction
InstructBioMol is a multimodal large language model designed for biomolecular instruction following. By integrating natural language with biomolecular data, InstructBioMol achieves any-to-any alignment between natural language, molecules, and proteins.

<div style="text-align:center">
  <img src="./framework.svg" width="85%"/>
</div>

### 🔧 Environment

The project requires the following two environments to run: (1) a training-inference environment and (2) a protein-molecule complex computation environment.
> **Note**: Please follow the recommended package versions when setting up the environment.

#### training-inference
First, create a new environment.
```bash
conda create --name biomol-train-infer python=3.8
```
Then, configure the environment according to the package details in `environment/train-infer.txt`.

#### protein-molecule complex

First, create a new environment.
```bash
conda create --name biomol-complex python=3.9
```
Configure the environment according to the requirements of [DiffDock](https://github.com/gcorso/DiffDock), and clone [DiffDock](https://github.com/gcorso/DiffDock) to local. Then, use the following command to install the other packages.
```bash
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
pip install meeko==0.1.dev3 vina==1.2.2 pdb2pqr==3.6.1
conda install -c conda-forge qvina openbabel
```

### 📚 Data

We provide the dataset in [Zenodo](https://zenodo.org/records/15303508). Please download the data to the project directory and use the following command to extract it:
```bash
mkdir data
cd data
unzip eval_assist.zip
unzip molecule-text.zip
unzip moledit.zip
unzip pdb-conf.zip
unzip protein-text.zip
unzip sdf.zip
unzip text2protmol.zip
```

The data also includes parameter files required for model execution. Please extract and save them to the `pretrained_ckpt` directory under the project root.
```bash
mkdir pretrained_ckpt
mv pretrained_ckpt.zip pretrained_ckpt/
cd pretraiend_ckpt
unzip pretrained_ckpt.zip
```

### 🤖  Pretrained Checkpoint


We release these variants of ​​InstructBioMol​​. Please download to the `pretrained_ckpt` directory.
| Model Name | Stage |  Multimodal| Description |
|------------|-----------| -------| -------|
| [InstructBioMol-base](https://huggingface.co/hicai-zju/InstructBioMol-base)   | Pretraining | ❎| Continual pretrained model on molecular sequences, protein sequences, and scientific literature. |
| [InstructBioMol-instruct-stage1](https://huggingface.co/hicai-zju/InstructBioMol-instruct-stage1) | Instruction tuning (stage 1) | ✅ |  Stage1 instruction-tuned model with biomolecular multimodal processing capabilities. (e.g., 3D molecules/proteins) |
| [InstructBioMol-instruct](https://huggingface.co/hicai-zju/InstructBioMol-instruct) |  Instruction tuning (stage 1 and 2) |  ✅| Fully instruction-tuned model (stage1 & stage2) with biomolecular multimodal processing capabilities (e.g., 3D molecules/proteins) |


### 🌟 Overview

The overall directory structure of the project is as follows:
```
├── 📂 code/                            # source code
├── 📂 config/                          # training & inference config 
├── 📂 data/                            # datasets
│   ├── 📂 molecule-text/               # datasets for aligning molecules with natural language
│   ├── 📂 protein-text/                # datasets for aligning proteins with natural language
│   ├── 📂 protein-molecule/            # datasets for aligning molecules with proteins
│   ├── 📂 pdb-conf/                    # protein structure files
│   ├── 📂 sdf/                         # molecule structure files
│   ├── 📂 eval-assist/                 # data for assisting evaluation
│   ├── 📂 moledit/                     # datasets for molecule editing
│   └── 📂 text2protmol/                # datasets for generating proteins and molecules conditioned on natural language descriptions
├──  📂 pretrained_ckpt/                # store the pretrained checkpoints
│   ├── 📂 InstructBioMol-base/         # InstructBioMol model
│   ├── 📂 InstructBioMol-instruct/     # InstructBioMol model
│   ├── 📂 esm2_t12_35M_UR50D/          # multimodal encoder parameter
│   ├── 📂 SaProt_35M_AF2/              # multimodal encoder parameter
│   ├── 📜 geoformer.ckpt               # multimodal encoder parameter
└── └── 📜 supervised_contextpred.pth   # multimodal encoder parameter
```
### 🚀 Training

Model training is conducted on 8 80G NVIDIA H800 GPUs.

```bash
conda activate biomol-train-infer
export TOKENIZERS_PARALLELISM=false
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_addr 127.0.0.1 --master_port $MASTER_PORT code/train.py \
                            --random_seed 0 \
                            --total_steps 900000 \
                            --eval_step 50000 \
                            --warmup_step 2000 \
                            --exp_name train \
                            --exp_id instructiontuning \
                            --lr 1e-5 \
                            --bs_per_gpu 3 \
                            --gradient_accumulation_steps 1 \
```
### 🔆 Inference and Evaluation

The following are scripts for inference on various downstream tasks.
#### molecule captioning

```bash
conda activate biomol-train-infer
CUDA_VISIBLE_DEVICES=0 python code/eval.py \
                       --dataset_name molecule_to_text_chebi_test \
                       --load_ckpt_path_list pretrained_ckpt/InstructBioMol-instruct \
                       --exp_name evaluation-moltext \
                       --exp_id mol2text \
                       --generate_bs 2 \
                       --generate_num_beams 5 
```
#### description-based molecule generation

```bash
conda activate biomol-train-infer
CUDA_VISIBLE_DEVICES=0 python code/eval.py \
                       --dataset_name text_to_molecule_chebi_test \
                       --load_ckpt_path_list pretrained_ckpt/InstructBioMol-instruct \
                       --exp_name evaluation-moltext \
                       --exp_id text2mol \
                       --generate_bs 2 \
                       --generate_num_beams 5
```
#### protein property answering

```bash
conda activate biomol-train-infer
CUDA_VISIBLE_DEVICES=0 python code/eval.py \
                       --dataset_name protein_to_text_swissprot_test_name \
                       --load_ckpt_path_list pretrained_ckpt/InstructBioMol-instruct \
                       --exp_name evaluation_prottext-sample \
                       --exp_id prot2name \
                       --generate_top_p 0.1 \
                       --generate_bs 8 

CUDA_VISIBLE_DEVICES=0 python code/eval.py \
                       --dataset_name protein_to_text_swissprot_test_family \
                       --load_ckpt_path_list pretrained_ckpt/InstructBioMol-instruct \
                       --exp_name evaluation_prottext-sample \
                       --exp_id prot2fam \
                       --generate_top_p 0.1 \
                       --generate_bs 8

CUDA_VISIBLE_DEVICES=0 python code/eval.py \
                       --dataset_name protein_to_text_swissprot_test_loc \
                       --load_ckpt_path_list pretrained_ckpt/InstructBioMol-instruct \
                       --exp_name evaluation_prottext-sample \
                       --exp_id prot2loc \
                       --generate_top_p 0.1 \
                       --generate_bs 8 

CUDA_VISIBLE_DEVICES=0 python code/eval.py \
                       --dataset_name protein_to_text_swissprot_test_func \
                       --load_ckpt_path_list pretrained_ckpt/InstructBioMol-instruct \
                       --exp_name evaluation_prottext-sample \
                       --exp_id prot2func \
                       --generate_top_p 0.1 \
                       --generate_bs 8 
```
#### description-based protein generation

```bash
conda activate biomol-train-infer
CUDA_VISIBLE_DEVICES=0 python code/eval.py \
                       --dataset_name text_to_protein_swissprot_test \
                       --load_ckpt_path_list pretrained_ckpt/InstructBioMol-instruct \
                       --exp_name evaluation_prottext-sample \
                       --exp_id text2protein \
                       --generate_top_p 0.9 \
                       --generate_t 0.8 \
                       --generate_bs 8
```

#### protein-based drug discovery

In this task, inference and evaluation are divided into the following steps:
1. Generate molecules based on the target proteins.
```bash
conda activate biomol-train-infer
CUDA_VISIBLE_DEVICES=0 python code/eval.py \
                       --dataset_name protein_to_molecule_bindingdb_test \
                       --eval_mode 2 \
                       --generate_N 100 \
                       --generate_n 25 \
                       --load_ckpt_path_list pretrained_ckpt/InstructBioMol-instruct \
                       --exp_name evaluation \
                       --exp_id protein2mol \
                       --generate_top_p 1 \
                       --generate_bs 1
```
2. Based on the generated molecules, use DiffDock to estimate the complex structure.
```bash
conda activate biomol-complex
python code/eval_gen_complex.py --data_file data_file --diffdock_path diffdock_path --mode p2m --gpu 0 --exp_id protein2mol 
```
`data_file` is the JSON file generated in the previous step, and `diffdock_path` is the directory where DiffDock is located.

3. Compute Vina Score based on complex structures.
```bash
conda activate biomol-complex
python code/eval_vina.py --folder generation --exp_id protein2mol --mode p2m
```
`generation` is the path to the folder named "generation" created in the second step.

#### substrate-based enzyme design

In this task, inference and evaluation are divided into the following steps:
1. Generate proteins for the target substrates.
```bash
conda activate biomol-train-infer
CUDA_VISIBLE_DEVICES=0 python code/eval.py \
                       --dataset_name molecule_to_protein_gorhea_test \
                       --eval_mode 2 \
                       --generate_N 100 \
                       --generate_n 10 \
                       --load_ckpt_path_list pretrained_ckpt/InstructBioMol-instruct \
                       --exp_name evaluation \
                       --exp_id mol2protein \
                       --generate_top_p 0.9 \
                       --generate_t 0.8 \
                       --generate_bs 1
```
2. Based on the generated proteins, use DiffDock to estimate the complex structure.
```bash
conda activate biomol-complex
python code/eval_gen_complex.py --data_file data_file --diffdock_path diffdock_path --mode m2p --gpu 0 --exp_id mol2protein
```
`data_file` is the JSON file generated in the previous step, and `diffdock_path` is the directory where DiffDock is located.

3. Compute ESP Score
```bash
conda activate biomol-train-infer
python code/eval_esp.py --data_file data_file --exp_id mol2protein
```
`data_file` is the JSON file generated in the first step.

4. Compute Vina Score based on complex structures.
```bash
conda activate biomol-complex
python code/eval_vina.py --folder generation --exp_id mol2protein --mode m2p
```
`generation` is the path to the folder named "generation" created in the second step.

### 📝 LoRA Fine-tuning

We also provide implementations for fine-tuning using LoRA on other tasks, including:
1. generating proteins and molecules simultaneously based on textual descriptions.
2. molecule editing tasks introduced in [ChatDrug](https://github.com/chao1224/ChatDrug).

#### generating proteins and molecules simultaneously based on textual descriptions
- training
```bash
conda activate biomol-train-infer
export TOKENIZERS_PARALLELISM=false

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
deepspeed --include localhost:0,1 --master_addr 127.0.0.1 --master_port $MASTER_PORT code/train_text2mp.py \
                            --random_seed 0 \
                            --total_steps 10000 \
                            --eval_step 5000 \
                            --warmup_step 2000 \
                            --exp_name train-text2pm \
                            --exp_id lora \
                            --dataset_name_list text_2_protein_molecule_train \
                            --dataset_selected_prob 1 \
                            --load_ckpt_path_list pretrained_ckpt/InstructBioMol-instruct \
                            --lr 1e-5 \
                            --bs_per_gpu 1 \
                            --gradient_accumulation_steps 2 \
                            --lora
```

- inference
```bash
CUDA_VISIBLE_DEVICES=0 python code/eval_text2mp.py \
                       --dataset_name text_2_protein_molecule_test \
                       --load_lora_ckpt_path_list pretrained_ckpt/InstructBioMol-text2mp \
                       --load_ckpt_path_list pretrained_ckpt/InstructBioMol-instruct \
                       --exp_name generation-text2pm \
                       --random_seed 1 \
                       --exp_id r1-p0.7 \
                       --generate_top_p 0.7 \
                       --generate_bs 1 \
                       --generate_N 10 \
                       --generate_n 5 \
                       --generate_max_new_tokens 512 \
                       --lora
```

#### molecule editing
- training
```bash
conda activate biomol-train-infer
export TOKENIZERS_PARALLELISM=false

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
deepspeed --include localhost:0,1 --master_addr 127.0.0.1 --master_port $MASTER_PORT code/train_moledit.py \
                            --random_seed 0 \
                            --total_steps 150000 \
                            --eval_step 10000 \
                            --warmup_step 2000 \
                            --exp_name Mol-Edit \
                            --exp_id train-all \
                            --dataset_name_list moledit_101 moledit_102 moledit_103 moledit_104 moledit_105 moledit_106 moledit_107 moledit_108 moledit_201 moledit_202 moledit_203 moledit_204 moledit_205 moledit_206 \
                            --dataset_selected_prob 1 1 1 1 1 1 1 1 1 1 1 1 1 1 \
                            --load_ckpt_path_list pretrained_ckpt/InstructBioMol-instruct \
                            --lr 1e-5 \
                            --bs_per_gpu 8 \
                            --gradient_accumulation_steps 1 \
                            --lora \
```

- inference
```bash
mode_list=(0 1)
task_list=(101 102 103 104 105 106 107 108 201 202 203 204 205 206)

for mode in "${mode_list[@]}"; do
for task in "${task_list[@]}"; do

CUDA_VISIBLE_DEVICES=0 python code/eval_moledit.py \
                       --dataset_name moledit_test_${task} \
                       --load_ckpt_path_list pretrained_ckpt/InstructBioMol-instruct \
                       --load_lora_ckpt_path_list pretrained_ckpt/InstructBioMol-moledit \
                       --exp_name generation-moledit \
                       --random_seed 0 \
                       --exp_id ${task}-mode${mode} \
                       --task_id ${task} \
                       --thres_mode ${mode} \
                       --generate_top_p 0.9 \
                       --generate_bs 10 \
                       --generate_N 1 \
                       --generate_n 1 \
                       --generate_max_new_tokens 512 \
                       --lora
done
done
```

### 🛠️ Utility Scripts

We provide utility scripts to preprocess custom data into model-ready formats:

Molecule Motif Extraction​
```
python utils/gen_molecule_motif.py
```

Protein FoldSeek Sequence Generation​
```
python utils/gen_foldseek_seq.py
```

Protein Motif Extraction​
```
python utils/gen_protein_motif.py
```

We also provide a utility script to extract ​​text-only parameters​​ from the InstructBioMol-instruct model, compatible with HuggingFace `LlamaForCausalLM`.
```
python utils/extract_base_params.py
```

### 🌻 Acknowledgement
We gratefully acknowledge the use of code from the following projects: [Geoformer](https://github.com/microsoft/AI2BMD/tree/Geoformer), [SaProt](https://github.com/westlake-repl/SaProt), [NExT-GPT](https://github.com/NExT-GPT/NExT-GPT), [MolT5](https://github.com/blender-nlp/MolT5), and [ESP](https://github.com/AlexanderKroll/ESP). Our work builds upon their foundational contributions.

### 🔖 Citation
```bibtex
@article{DBLP:journals/corr/abs-2410-07919,
  author       = {Xiang Zhuang and
                  Keyan Ding and
                  Tianwen Lyu and
                  Yinuo Jiang and
                  Xiaotong Li and
                  Zhuoyi Xiang and
                  Zeyuan Wang and
                  Ming Qin and
                  Kehua Feng and
                  Jike Wang and
                  Qiang Zhang and
                  Huajun Chen},
  title        = {InstructBioMol: Advancing Biomolecule Understanding and Design Following
                  Human Instructions},
  journal      = {CoRR},
  volume       = {abs/2410.07919},
  year         = {2024}
}
```

### 😀 About
If you have any questions, please contact Mr. Xiang Zhuang at zhuangxiang@zju.edu.cn.