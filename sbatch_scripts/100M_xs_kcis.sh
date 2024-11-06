#!/bin/bash
#SBATCH --partition=lovelace
#SBATCH --account=kcis
#SBATCH --qos=kl1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --mem=100000
#SBATCH --time=4-00:00:00
#SBATCH --output=Stdout_xs.txt
#SBATCH --mail-user=saketh.vemula@research.iiit.ac.in
#SBATCH --mail-type=ALL


source /home2/saketh.vemula/ltg_venv/bin/activate

# Set environment variables
#export WORLD_SIZE=4
#export SLURM_GPUS_ON_NODE=4

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
    train.py --input_path "../data/processed_100M/cached_{sequence_length}.txt" --input_valid_path "../data/processed_dev/cached_{sequence_length}.txt" --validation_freq 10000 --validation_max_steps 1000 --config_file "../configs/xs.json" --output_dir "../checkpoints/xs" --vocab_path "../tokenizer.json" --batch_size 400 --max_steps 250000 --learning_rate 0.01 --long_after 1.0

#python3 train.py --input_path "../data/processed_10M/all.txt" --config_file "../configs/xs.json" --output_dir "../checkpoints/xs" --vocab_path "../tokenizer.json" --batch_size 64 --max_steps 500000 --long_after 0.9


#srun python3 -u train.py --input_path "../data/processed_10M/all.txt" --config_file "../configs/xs.json" --output_dir "../checkpoints/xs" --vocab_path "../tokenizer.json" --batch_size 64 --max_steps 500000 --long_after 0.9
