import wandb
import torch
import os
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from model import BertForSequenceClassification
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import random
from tokenizers import Tokenizer
from dataclasses import dataclass
from typing import Optional

# Set wandb keyc
os.environ["WANDB_API_KEY"] = "c8a7a539cb5fed3df89b21d71956ca6b4befd2a5"

# Configurations
LANG = "te"
TOKENIZER_TYPE = "M-BPE"
VOCAB_SIZE = 16384
CONFIG_FILE = "/home2/saketh.vemula/babylm/configs/xs.json"
TOKENIZER_PATH = "/home2/saketh.vemula/babylm/tokenizers/te/te_M-BPE_16384.json"
CHECKPOINT_PATH = "/home2/saketh.vemula/te_xs_BPE_M-16384.bin"
PATH_TO_TEST_DATA = "~/data_Test/te/"
SET = "all" # all / IOV / OOV
NUM_EPOCHS = 20
LEARNING_RATE = 1e-5
BATCH_SIZE = 64
WEIGHT_DECAY = 0.01
MAX_SEQ_LEN = 32

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_data(file_path, is_test=False):
    df = pd.read_csv(file_path, delimiter='\t', header=None, 
                     names=['word', 'morphology', 'label'], skiprows=1)
    if not is_test:
        # Only drop NaN for training and validation data
        df.dropna(inplace=True)
    return df

def encode_pair(tokenizer, text_a, text_b, max_length=MAX_SEQ_LEN):
    encoded = tokenizer.encode(text_a, text_b)
    if len(encoded.ids) > max_length:
        encoded.truncate(max_length)
    else:
        encoded.pad(max_length)
    return {
        'input_ids': torch.tensor(encoded.ids, dtype=torch.long).unsqueeze(0),
        'attention_mask': torch.tensor(encoded.attention_mask, dtype=torch.long).unsqueeze(0)
    }

def prepare_data(words_a, words_b, labels=None, tokenizer=None, is_test=False):
    input_ids = []
    attention_masks = []
    
    for sent in zip(words_a, words_b):
        encoded_dict = encode_pair(tokenizer, sent[0], sent[1])
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    if is_test:
        # For test data, create dummy labels if None
        if labels is None:
            labels = torch.zeros(len(input_ids), dtype=torch.long)
        else:
            # Convert NaN to -1 for test set labels
            labels = torch.tensor([l if not pd.isna(l) else -1 for l in labels], dtype=torch.long)
    else:
        labels = torch.tensor(labels, dtype=torch.long)
    
    return input_ids, attention_masks, labels

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    # Only consider non-negative labels (ignore -1 labels from test set)
    mask = labels_flat >= 0
    return np.sum(pred_flat[mask] == labels_flat[mask]) / max(mask.sum(), 1)

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

@dataclass
class TrainingConfig:
    # Model configurations
    config_file: str = CONFIG_FILE
    tokenizer_path: str = TOKENIZER_PATH
    checkpoint_path: str = CHECKPOINT_PATH
    
    # Data configurations
    data_path: str = PATH_TO_TEST_DATA
    max_seq_len: int = MAX_SEQ_LEN
    
    # Training hyperparameters
    num_epochs: int = NUM_EPOCHS
    learning_rate: float = LEARNING_RATE
    batch_size: int = BATCH_SIZE
    weight_decay: float = WEIGHT_DECAY
    warmup_steps: int = 0
    grad_clip: float = 1.0
    
    # Other settings
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Wandb configurations
    wandb_project: str = "Fine_Tuning"
    wandb_entity: Optional[str] = "vemulasakethreddy_10"
    wandb_run_name: Optional[str] = f"WaW_{LANG}_xs_{TOKENIZER_TYPE}_{VOCAB_SIZE}"

def init_wandb(config: TrainingConfig):
    """Initialize wandb with the given configuration."""
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        config={
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "weight_decay": config.weight_decay,
            "warmup_steps": config.warmup_steps,
            "max_seq_len": config.max_seq_len,
            "grad_clip": config.grad_clip,
            "seed": config.seed,
            "device": config.device,
            "model_config": config.config_file,
            "checkpoint_path": config.checkpoint_path,
        }
    )

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, train_dataloader, optimizer, scheduler, device, epoch):
    total_train_loss = 0
    model.train()
    t0 = time.time()
    
    for step, batch in enumerate(train_dataloader):
        if step % 10 == 0 and step != 0:
            elapsed = format_time(time.time() - t0)
            print(f'  Batch {step:>5,}  of  {len(train_dataloader):>5,}.    Elapsed: {elapsed}.')
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        model.zero_grad()
        
        result = model(b_input_ids,
                      attention_mask=b_input_mask,
                      labels=b_labels,
                      return_dict=True)
        
        loss = result['loss']
        total_train_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), wandb.config.grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        # Log batch-level metrics
        wandb.log({
            "batch_loss": loss.item(),
            "learning_rate": scheduler.get_last_lr()[0],
            "batch": step + len(train_dataloader) * epoch
        })
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    return avg_train_loss

def evaluate(model, dataloader, device, is_test=False):
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    predictions = []
    all_labels = []
    
    for batch in dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():
            result = model(b_input_ids,
                         attention_mask=b_input_mask,
                         labels=b_labels,
                         return_dict=True)
        
        loss = result['loss']
        logits = result['logits']
        
        total_eval_loss += loss.item()
        
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()
        
        predictions.extend(np.argmax(logits, axis=1))
        all_labels.extend(label_ids)
        
        total_eval_accuracy += flat_accuracy(logits, label_ids)
    
    avg_accuracy = total_eval_accuracy / len(dataloader)
    avg_loss = total_eval_loss / len(dataloader)
    
    # Calculate confusion matrix and other metrics
    valid_mask = np.array(all_labels) >= 0
    true_labels = np.array(all_labels)[valid_mask]
    pred_labels = np.array(predictions)[valid_mask]
    
    if len(true_labels) > 0:
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=true_labels,
                preds=pred_labels,
                class_names=["0", "1"]
            )
        })
    
    if is_test:
        return avg_accuracy, avg_loss, predictions
    return avg_accuracy, avg_loss

def main():
    config = TrainingConfig()
    set_seed(config.seed)
    init_wandb(config)
    
    # Rest of the data loading code remains the same...

    # Load data
    if SET == "IOV":
        train_df = load_data(f"{PATH_TO_TEST_DATA}WaW_{LANG}_IOV_train.tsv")
        val_df = load_data(f"{PATH_TO_TEST_DATA}WaW_{LANG}_IOV_dev.tsv")
        test_df = load_data(f"{PATH_TO_TEST_DATA}WaW_{LANG}_IOV_test.tsv", is_test=True)
    elif SET == "OOV":
        train_df = load_data(f"{PATH_TO_TEST_DATA}WaW_{LANG}_OOV_train.tsv")
        val_df = load_data(f"{PATH_TO_TEST_DATA}WaW_{LANG}_OOV_dev.tsv")
        test_df = load_data(f"{PATH_TO_TEST_DATA}WaW_{LANG}_OOV_test.tsv", is_test=True)
    elif SET == "all":
        train_df = load_data(f"{PATH_TO_TEST_DATA}WaW_{LANG}_all_train.tsv")
        val_df = load_data(f"{PATH_TO_TEST_DATA}WaW_{LANG}_all_dev.tsv")
        test_df = load_data(f"{PATH_TO_TEST_DATA}WaW_{LANG}_all_test.tsv", is_test=True)
    elif SET == "IOV_train_OOV_train_IOV_test":
        train_df = load_data(f"{PATH_TO_TEST_DATA}WaW_{LANG}_IOV_train.tsv")
        val_df = load_data(f"{PATH_TO_TEST_DATA}WaW_{LANG}_OOV_train.tsv")
        test_df = load_data(f"{PATH_TO_TEST_DATA}WaW_{LANG}_IOV_test.tsv", is_test=True)
    elif SET == "none":
        train_df = load_data(f"{PATH_TO_TEST_DATA}WaW.train.tsv")
        val_df = load_data(f"{PATH_TO_TEST_DATA}WaW.dev.tsv")
        test_df = load_data(f"{PATH_TO_TEST_DATA}WaW.test.tsv", is_test=True)
    
    print(f"Number of Training instances: {train_df.shape[0]}")
    print(f"Number of Validation instances: {val_df.shape[0]}")
    print(f"Number of Test instances: {test_df.shape[0]}")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    # Prepare datasets
    train_inputs = prepare_data(train_df.word.values, train_df.morphology.values, 
                              train_df.label.values, tokenizer)
    val_inputs = prepare_data(val_df.word.values, val_df.morphology.values, 
                            val_df.label.values, tokenizer)
    test_inputs = prepare_data(test_df.word.values, test_df.morphology.values, 
                             test_df.label.values, tokenizer, is_test=True)
    
    # Create dataloaders
    train_dataset = TensorDataset(*train_inputs)
    val_dataset = TensorDataset(*val_inputs)
    test_dataset = TensorDataset(*test_inputs)
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    validation_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = BertForSequenceClassification(
        config_path=CONFIG_FILE,
        vocab_path=TOKENIZER_PATH,
        num_labels=2,
        model_path=CHECKPOINT_PATH
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8, 
                           weight_decay=WEIGHT_DECAY)
    total_steps = len(train_dataloader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=0,
                                              num_training_steps=total_steps)
    
    # Training loop
    training_stats = []
    total_t0 = time.time()
    
    # Training loop
    for epoch_i in range(config.num_epochs):
        print(f'\n======== Epoch {epoch_i + 1} / {config.num_epochs} ========')
        print('Training...')
        t0 = time.time()
        
        avg_train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, 
                                   config.device, epoch_i)
        training_time = format_time(time.time() - t0)
        
        print("\nRunning Validation...")
        t0 = time.time()
        avg_val_accuracy, avg_val_loss = evaluate(model, validation_dataloader, config.device)
        validation_time = format_time(time.time() - t0)
        
        # Log epoch-level metrics
        wandb.log({
            "epoch": epoch_i + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": avg_val_accuracy,
            "train_time": time.time() - t0
        })
        
        # Log model gradients and parameters
        # wandb.watch(model, log="all")
    
    # # Final evaluation on test set
    # print("\nRunning final evaluation on test set...")
    # test_accuracy, test_loss, predictions = evaluate(model, test_dataloader, 
    #                                                config.device, is_test=True)
    
    # # Log final test metrics
    # wandb.log({
    #     "test_accuracy": test_accuracy,
    #     "test_loss": test_loss
    # })
    
    # # Save the model
    # model_artifact = wandb.Artifact(
    #     name=f"model-{wandb.run.id}",
    #     type="model",
    #     description="Fine-tuned WaW model"
    # )
    # torch.save(model.state_dict(), "model.pt")
    # model_artifact.add_file("model.pt")
    # wandb.log_artifact(model_artifact)
    
    # # Save predictions
    # test_df['predicted'] = predictions
    # predictions_path = f"{config.data_path}WaW.test.predictions.tsv"
    # test_df.to_csv(predictions_path, sep='\t', index=False)
    
    # # Log predictions as an artifact
    # predictions_artifact = wandb.Artifact(
    #     name=f"predictions-{wandb.run.id}",
    #     type="predictions",
    #     description="Model predictions on test set"
    # )
    # predictions_artifact.add_file(predictions_path)
    # wandb.log_artifact(predictions_artifact)
    
    wandb.finish()

if __name__ == "__main__":
    main()
