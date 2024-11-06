#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=24:00:00
#SBATCH --output=eng_xs_BPE_v50227.txt
#SBATCH --mail-user=saketh.vemula@research.iiit.ac.in
#SBATCH --mail-type=ALL

# Set environment variables
#export WORLD_SIZE=4
#export SLURM_GPUS_ON_NODE=4

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
	train.py --input_path "../data/en/cached_{sequence_length}_train.txt" --input_valid_path "../data/en/cached_{sequence_length}_dev.txt" --validation_freq 5000 --validation_max_steps 500 --config_file "../configs/xs_v50227.json" --output_dir "../checkpoints/xs" --vocab_path "../tokenizers/en/en_BPE_50227.json" --batch_size 400 --max_steps 100000 --learning_rate 0.01 --long_after 1.0

#python3 train.py --input_path "../data/processed_10M/all.txt" --config_file "../configs/xs.json" --output_dir "../checkpoints/xs" --vocab_path "../tokenizer.json" --batch_size 64 --max_steps 500000 --long_after 0.9


#srun python3 -u train.py --input_path "../data/processed_10M/all.txt" --config_file "../configs/xs.json" --output_dir "../checkpoints/xs" --vocab_path "../tokenizer.json" --batch_size 64 --max_steps 500000 --long_after 0.9

