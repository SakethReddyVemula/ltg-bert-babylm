#!/bin/bash
#SBATCH --partition=lovelace
#SBATCH --account=kcis
#SBATCH --qos=kl1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --mem=100000
#SBATCH --time=1-10:00:00
#SBATCH --output=en_xs_BPE_v16384.txt
#SBATCH --mail-user=saketh.vemula@research.iiit.ac.in
#SBATCH --mail-type=ALL

source /home2/saketh.vemula/ltg_venv/bin/activate
export WANDB_API_KEY="c8a7a539cb5fed3df89b21d71956ca6b4befd2a5" # Set api key of wandb in script

get_free_port() {
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); port = s.getsockname()[1]; s.close(); print(port)"
}

export MASTER_PORT=$(get_free_port)
echo "MASTER_PORT="$MASTER_PORT

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# Run the training with torchrun
torchrun --nnodes=1 --nproc_per_node=1 \
    train_wo_val.py --input_path "/home2/saketh.vemula/babylm/data/en/cached_128_train.txt" --input_valid_path "/home2/saketh.vemula/babylm/data/en/cached_128_dev.txt" --validation_freq 10000 --validation_max_steps 2000 --config_file "/home2/saketh.vemula/babylm/configs/xs.json" --output_dir "/home2/saketh.vemula/babylm/checkpoints/xs/en/" --vocab_path "/home2/saketh.vemula/babylm/tokenizers/en/en_BPE_16384.json" --batch_size 1400 --max_steps 100000 --learning_rate 0.01 --long_after 1.0

