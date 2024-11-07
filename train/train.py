# coding=utf-8

import os
import os.path
import argparse
from tqdm import tqdm
from itertools import count
from socket import gethostname

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from tokenizers import Tokenizer
from lamb import Lamb
from config import BertConfig

from model import Bert

from utils import cosine_schedule_with_warmup, is_main_process, get_rank, seed_everything, get_world_size
from dataset import Dataset
import sys 
import subprocess 

# 1. Set MASTER_ADDR to the first node in the SLURM job node list ###
#if 'SLURM_JOB_NODELIST' in os.environ: ###
#    master_addr = subprocess.getoutput("scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1") ###
#    os.environ['MASTER_ADDR'] = master_addr ###

# 2. Set MASTER_PORT to a fixed value or choose a random free port (optional) ###
# os.environ['MASTER_PORT'] = '12345' ###

if int(os.environ["SLURM_PROCID"]) == 0:
    import wandb
#    os.environ["WANDB_API_KEY"] = "" # set this if not there in script


# Set WORLD_SIZE
#if "WORLD_SIZE" not in os.environ:
os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())

# Set SLURM_GPUS_ON_NODE
if "SLURM_GPUS_ON_NODE" not in os.environ:
    os.environ["SLURM_GPUS_ON_NODE"] = str(torch.cuda.device_count())

# Print the environment variables for verification
print(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}")
print(f"SLURM_GPUS_ON_NODE: {os.environ['SLURM_GPUS_ON_NODE']}")
print(f"SLURM_PROCID: {os.environ['SLURM_PROCID']}")


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_path", default="../data/processed/cached_{sequence_length}.txt.gz", type=str, help="The input data dir. Should contain .hdf5 files for the task.")
    parser.add_argument("--config_file", default="../configs/base.json", type=str, help="The BERT model config")
    parser.add_argument("--output_dir", default="../checkpoints/base", type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--vocab_path", default="../tokenizer.json", type=str, help="The vocabulary the BERT model will train on.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to a previous checkpointed training state.")
    parser.add_argument("--input_valid_path", default="../data/processed/cached_valid_{sequence_length}.txt.gz", type=str, help="The input validation data dir. Should contain .hdf5 files for validation.") #@
    parser.add_argument("--validation_freq", default=1000, type=int, help="Frequency of validation in steps.") #@
    parser.add_argument("--validation_max_steps", default=100, type=int, help="Total number of validation steps to be performed on validation dataset") #@

    # Other parameters
    parser.add_argument("--optimizer", default="lamb", type=str)
    parser.add_argument("--scheduler", default="cosine", type=str)
    parser.add_argument("--seq_length", default=128, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=1024, type=int, help="Total batch size for training per GPUs and per grad accumulation step.")
    parser.add_argument("--learning_rate", default=1.0e-2, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_steps", default=31250 // 2, type=int, help="Total number of training steps to perform.")
    parser.add_argument("--long_after", default=0.9, type=float) 
    parser.add_argument("--warmup_proportion", default=0.016, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--log_freq', type=int, default=10, help='frequency of logging loss.')
    parser.add_argument("--mask_p", default=0.15, type=float, help="Masking probability.")
    parser.add_argument("--short_p", default=0.1, type=float, help="Short sequence probability.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Short sequence probability.")
    parser.add_argument("--max_gradient", default=2.0, type=float, help="Max value for gradient clipping.")
    parser.add_argument('--mixed_precision', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--activation_checkpointing', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    print("parsed the arguments") 
    return args


@torch.no_grad()
def log_parameter_histograms(model, step):
    for name, param in model.named_parameters():
        wandb.log(
            {
                f"parameters/norm_{name}": torch.linalg.norm(param.data).cpu().item(),
                f"parameters/std_{name}": param.data.std().cpu().item(),
            },
            step=step,
            commit=False
        )
        if param.requires_grad and param.grad is not None:
            wandb.log(
                {
                    f"gradients/norm_{name}": torch.linalg.norm(param.grad).cpu().item(),
                    f"gradients/std_{name}": param.grad.std().cpu().item(),
                },
                step=step,
                commit=False
            )

def setup_training(args):
    print("setting-up training")
    assert torch.cuda.is_available()
    args.n_gpu = torch.cuda.device_count()

    if "MASTER_ADDR" not in os.environ:
        raise ValueError("MASTER_ADDR environment variable is not set")
    if "MASTER_PORT" not in os.environ:
        raise ValueError("MASTER_PORT environment variable is not set")

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    gpus_per_node = args.n_gpu
    
    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)

    print("seeding everything")
    seed_everything(args.seed + rank)
    print("seeding done")

    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    if rank == 0:
        print(f"Group initialized? {torch.distributed.is_initialized()}", flush=True)

    print("Initialized the distributed training environment using NCCL as the communication backend")
    
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print(f"NCCL started on device {device}", flush=True)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    if is_main_process():
        os.system(f"mkdir -p {args.output_dir}")
        print("args.output_dir created using mkdir")

    if is_main_process():
        print(f"Training for {args.max_steps:,} steps with {get_world_size()} GPUs")
        print(f"In total, the model will be trained on 'steps'({args.max_steps:,}) x 'GPUs'({get_world_size()}) x 'batch_size'({args.batch_size:,}) x 'seq_len'({args.seq_length:,}) = {args.max_steps * get_world_size() * args.batch_size * args.seq_length:,} subword instances")

    args.device_max_steps = args.max_steps

    if is_main_process():
        wandb.init(
            name="LTG-BERT base 100M",
            config=args,
            id=args.wandb_id,
            project="BABY-LM",
            entity="vemulasakethreddy_10"
        )

    print("setting-up training done")
    return device, local_rank

def prepare_model_and_optimizer(args, device, local_rank, checkpoint):
    config = BertConfig(args.config_file)
    model = Bert(config, args.activation_checkpointing)

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update(config.to_dict())
        wandb.config.update({"n_params": n_params})
        print(model)
        print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=False)

    model.to(device)

    no_decay = ['bias', 'layer_norm', 'embedding']
    decay_params = [(n, p) for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    no_decay_params = [(n, p) for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    optimizer_grouped_parameters = [
        {'params': [p for _, p in decay_params], 'weight_decay': args.weight_decay},
        {'params': [p for _, p in no_decay_params], 'weight_decay': 0.0}
    ]

    if is_main_process():
        print("Parameters without weight decay:")
        for n, _ in no_decay_params:
            print(n)
        print()
        print("Parameters with weight decay:")
        for n, _ in decay_params:
            print(n)
        print(flush=True)

    if args.optimizer == "adam" or args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )
    elif args.optimizer == "lamb":
        optimizer = Lamb(
            optimizer_grouped_parameters,
            args.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

    scheduler = cosine_schedule_with_warmup(optimizer, int(args.device_max_steps * args.warmup_proportion), args.device_max_steps, 0.1)
    
    print("scheduler assigned") 
    
    #model = DistributedDataParallel(
    #    model,
    #    device_ids=[local_rank],
    #    bucket_cap_mb=torch.cuda.get_device_properties(device).total_memory,
    #    broadcast_buffers=False,
    #    gradient_as_bucket_view=True,
    #    static_graph=True
    #)

    #grad_scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    #if checkpoint is not None:
    #    optimizer.load_state_dict(checkpoint["optimizer"])
    #    scheduler.load_state_dict(checkpoint["scheduler"])
    #    grad_scaler.load_state_dict(checkpoint["grad_scaler"])

    print(f"[Rank {get_rank()}] About to initialize DDP", flush=True)
    torch.distributed.barrier()  # Ensure all processes are here before DDP init

    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        # Remove or reduce bucket_cap_mb
        broadcast_buffers=False,
        # Remove gradient_as_bucket_view and static_graph or set to False
    )

    print(f"[Rank {get_rank()}] DDP initialized", flush=True)
    torch.distributed.barrier()  # Ensure all processes completed DDP init

    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        grad_scaler.load_state_dict(checkpoint["grad_scaler"])

    print(f"[Rank {get_rank()}] prepare_model_and_optimizer done!", flush=True)
    return model, config, optimizer, scheduler, grad_scaler


@torch.no_grad()
def evaluate(model, data, valid_dataloader, args, device, global_step):
    model.eval()
    # valid_dataloader = create_train_dataloader(data, args, global_step, args.seed + get_rank())
    
    total_loss = 0
    total_accuracy = 0
    total_samples = 0

    if is_main_process():
        valid_iter = tqdm(valid_dataloader, desc="Valid iteration", initial=0, total=args.validation_max_steps)
    else:
        valid_iter = valid_dataloader

    for local_step, batch in enumerate(valid_iter):
        input_ids, attention_mask, target_ids = [t.to(device, non_blocking=True) for t in batch]
        input_ids, target_ids = input_ids.t(), target_ids.t()

        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
            prediction = model(input_ids, attention_mask, target_ids)

            target_ids = target_ids.flatten()
            target_ids = target_ids[target_ids != -100]
            loss = F.cross_entropy(prediction, target_ids)
            perplexity = torch.exp(loss)
        
        with torch.no_grad():
            accuracy = (prediction.argmax(-1) == target_ids).float().mean()

        valid_iter.set_postfix_str(f"loss: {loss.item():.2f}, accuracy: {accuracy.item() * 100.0:.2f}, perplexity: {perplexity.item():.2f}")

        total_loss += loss.item() * len(target_ids)
        total_accuracy += accuracy.item() * len(target_ids)
        # total_perplexity += perplexity.item() * len(target_ids)
        total_samples += len(target_ids)

        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples
        # avg_perplexity = total_perplexity / total_samples

        if is_main_process():
            valid_iter.set_postfix_str(f"loss: {loss.item():.2f}, accuracy: {accuracy.item() * 100.0:.2f}, perplexity: {perplexity.item():.2f}")
            wandb.log(
                {
                    "valid/loss": avg_loss,
                    "valid/accuracy": avg_accuracy * 100.0,
                    "valid/perplexity": perplexity
                },
                step=global_step,
            )
            # print(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy * 100.0:.2f}%")

        # Exiting the training due to hitting max steps
        if local_step >= args.validation_max_steps:
            print("Evaluation done\n")
            return avg_loss, avg_accuracy, perplexity

    print("Evaluation done\n")

    return avg_loss, avg_accuracy, perplexity


def training_epoch(model, data, valid_data, optimizer, scheduler, grad_scaler, global_step, epoch, args, device, max_local_steps, valid_max_local_steps):
    train_dataloader = create_train_dataloader(data, args, global_step, args.seed + get_rank() + epoch * get_world_size())
    valid_dataloader = create_train_dataloader(valid_data, args, global_step, args.seed + get_rank() + epoch * get_world_size()) #@
    print("train_dataloader ready") 
    print("valid_dataloader ready") 

    model = model.train()
    optimizer.zero_grad(set_to_none=True)

    if is_main_process():
        train_iter = tqdm(train_dataloader, desc="Train iteration", initial=global_step, total=args.device_max_steps)
    else:
        train_iter = train_dataloader

    for local_step, batch in enumerate(train_iter):
        input_ids, attention_mask, target_ids = [t.to(device, non_blocking=True) for t in batch]
        input_ids, target_ids = input_ids.t(), target_ids.t()

        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
            prediction = model(input_ids, attention_mask, target_ids)

            target_ids = target_ids.flatten()
            target_ids = target_ids[target_ids != -100]
            loss = F.cross_entropy(prediction, target_ids)
            perplexity = torch.exp(loss)

        with torch.no_grad():
            accuracy = (prediction.argmax(-1) == target_ids).float().mean()

        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient)

        return_value = grad_scaler.step(optimizer)
        grad_scaler.update()

        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if return_value is None:
            continue

        scheduler.step()

        if is_main_process():
            train_iter.set_postfix_str(f"loss: {loss.item():.2f}, accuracy: {accuracy.item() * 100.0:.2f}, grad_norm: {grad_norm:.2f}, lr: {optimizer.param_groups[0]['lr']:.5f}")

            if global_step % 100 == 0:
                log_parameter_histograms(model, global_step)

            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": loss.item(),
                    "train/accuracy": accuracy.item() * 100.0,
                    "train/perplexity": perplexity.item(),
                    "stats/learning_rate": optimizer.param_groups[0]['lr'],
                    "stats/grad_norm": grad_norm,
                    "stats/seq_length": data.seq_length,
                    "stats/grad_scale": grad_scaler.get_scale()
                },
                step=global_step,
            )

        if global_step % args.validation_freq == 0:
            valid_loss, valid_accuracy, valid_perplexity = evaluate(model, valid_data, valid_dataloader, args, device, global_step)

        if global_step == int(args.device_max_steps * args.long_after):
            return global_step

        # Exiting the training due to hitting max steps
        if global_step >= args.device_max_steps or local_step >= max_local_steps - 1:
            return global_step
        
        

    return global_step


def save(model, optimizer, grad_scaler, scheduler, global_step, epoch, args):
    checkpoint_path = f"{args.output_dir}/model.bin"
    if is_main_process():
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
        torch.save(
            {
                "model": model_to_save.state_dict(),
                "optimizer": optimizer.state_dict(),
                "grad_scaler": grad_scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "args": args,
            },
            checkpoint_path
        )

    return checkpoint_path


def load_dataset(args, tokenizer, device):
    # seq_length = args.seq_length * 4 if global_step >= int(args.device_max_steps * args.long_after) else args.seq_length
    seq_length = args.seq_length
    train_data = Dataset(
        args.input_path.format(sequence_length=seq_length), get_rank(), get_world_size(), tokenizer, seq_length, args.mask_p, args.short_p
    )
    print(f"Loaded training file {get_rank()}", flush=True)

    # batch_size = args.batch_size // 4 if global_step > args.device_max_steps * args.long_after else args.batch_size
    batch_size = args.batch_size
    min_length = torch.tensor(len(train_data) // batch_size, dtype=torch.long, device=device)
    torch.distributed.all_reduce(min_length, torch.distributed.ReduceOp.MIN)

#    if is_main_process():
#        train_data.show_random_item()

    return train_data, min_length

#@ New function
def load_validation_dataset(args, tokenizer, device):
    # seq_length = args.seq_length * 4 if global_step >= int(args.device_max_steps * args.long_after) else args.seq_length
    seq_length = args.seq_length
    valid_data = Dataset(
        args.input_valid_path.format(sequence_length=seq_length), get_rank(), get_world_size(), tokenizer, seq_length, args.mask_p, args.short_p
    )
    print(f"Loaded validation file {get_rank()}", flush=True)

    # batch_size = args.batch_size // 4 if global_step > args.device_max_steps * args.long_after else args.batch_size
    batch_size = args.batch_size
    min_length = torch.tensor(len(valid_data) // batch_size, dtype=torch.long, device=device)
    torch.distributed.all_reduce(min_length, torch.distributed.ReduceOp.MIN)

    return valid_data, min_length


def create_train_dataloader(data, args, global_step, seed):
    # batch_size = args.batch_size // 4 if global_step >= int(args.device_max_steps * args.long_after) else args.batch_size
    batch_size = args.batch_size
    train_dataloader = DataLoader(
        data,
        shuffle=True,
        batch_size=batch_size,
        num_workers=7 - 1,
        generator=torch.Generator().manual_seed(seed),
        drop_last=True,
        pin_memory=True
    )
    return train_dataloader

if __name__ == "__main__":
    print("parsing arguments") 
    args = parse_arguments()
    print(f"max_steps: {args.max_steps}")
    print(f"learning_rate: {args.learning_rate}")
    new_max_steps = args.max_steps
    new_lr = args.learning_rate

    print("Loading Checkpoints")
    if args.checkpoint_path is not None:
        print("args.checkpoint_path provided") 
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_args, initial_epoch, global_step = checkpoint["args"], checkpoint["epoch"] + 1, checkpoint["global_step"]
        args = vars(args).copy()
        args.update(vars(checkpoint_args))
        args = argparse.Namespace(**args)

        args.max_steps = new_max_steps
        args.learning_rate = new_lr
        print("args.checkpoint_path successfully loaded") 
    else:
        print("args.checkpoint_path not provided") 
        checkpoint, initial_epoch, global_step = None, 0, 0
        args.wandb_id = wandb.util.generate_id() if int(os.environ["SLURM_PROCID"]) == 0 else 0


    print(f"global_steps: {global_step}")
    print(f"max_steps: {args.max_steps}")
    print(f"learning_rate: {args.learning_rate}")

    print("loading tokenizer from args.vocab_path") 
    tokenizer = Tokenizer.from_file(args.vocab_path)
    print("loaded tokenizer from args.vocab_path") 

    device, local_rank = setup_training(args)
    
    model, config, optimizer, scheduler, grad_scaler = prepare_model_and_optimizer(args, device, local_rank, checkpoint)

    train_data, min_length = load_dataset(args, tokenizer, device)
    valid_data, valid_min_length = load_validation_dataset(args, tokenizer, device) #@

    for epoch in count(initial_epoch):
        # if global_step == int(args.device_max_steps * args.long_after):
        #     train_data, min_length = load_dataset(args, tokenizer, device)
        #     valid_data, valid_min_length = load_validation_dataset(args, tokenizer, device) #@

        global_step = training_epoch(model, train_data, valid_data, optimizer, scheduler, grad_scaler, global_step, epoch, args, device, min_length, valid_min_length)
        checkpoint_path = save(model, optimizer, grad_scaler, scheduler, global_step, epoch, args)
        
        if global_step >= args.device_max_steps:
            break

