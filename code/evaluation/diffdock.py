import os


def call_diffdock_csv(csv_file, diffdock_path, dump_path, gpu):
    os.makedirs(dump_path, exist_ok=True)

    cmd = f'CUDA_VISIBLE_DEVICES={gpu} python {diffdock_path}/inference.py ' \
          f'--protein_ligand_csv {csv_file} ' \
          f'--out_dir {dump_path} ' \
          f'--config {diffdock_path}/default_inference_args.yaml'
    os.system(cmd)

