#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=96:00:00
#SBATCH --mincpus=40
#SBATCH --mail-user=saketh.vemula@research.iiit.ac.in
#SBATCH --mail-type=ALL

# Set environment variables
export WORLD_SIZE=4
export SLURM_GPUS_ON_NODE=4

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
torchrun --nnodes=1 --nproc_per_node=4 \
    train.py --input_path "../data/processed_10M/cached_{sequence_length}.txt" --config_file "../configs/xs.json" --output_dir "../checkpoints/xs" --vocab_path "../tokenizer_10M.json" --batch_size 64 --max_steps 5000 --long_after 0.9
#python3 train.py --input_path "../data/processed_10M/all.txt" --config_file "../configs/xs.json" --output_dir "../checkpoints/xs" --vocab_path "../tokenizer.json" --batch_size 64 --max_steps 500000 --long_after 0.9


#srun python3 -u train.py --input_path "../data/processed_10M/all.txt" --config_file "../configs/xs.json" --output_dir "../checkpoints/xs" --vocab_path "../tokenizer.json" --batch_size 64 --max_steps 500000 --long_after 0.9
