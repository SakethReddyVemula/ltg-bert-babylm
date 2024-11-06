# Import models
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

# Set default dtype
torch.set_default_dtype(torch.float32)

# Configurations
CONFIG_FILE = "/home2/saketh.vemula/babylm/configs/xs.json"
TOKENIZER_PATH = "/home2/saketh.vemula/babylm/tokenizers/en/en_BPE_16384.json"
CHECKPOINT_PATH = "/home2/saketh.vemula/en_xs_BPE_16384.bin"
PATH_TO_TEST_DATA = "~/data_Test/eng/"
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

def train_epoch(model, train_dataloader, optimizer, scheduler, device):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
    
    return total_train_loss / len(train_dataloader)

def evaluate(model, dataloader, device, is_test=False):
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    predictions = []
    
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
        
        if is_test:
            predictions.extend(np.argmax(logits, axis=1))
        
        total_eval_accuracy += flat_accuracy(logits, label_ids)
    
    avg_accuracy = total_eval_accuracy / len(dataloader)
    avg_loss = total_eval_loss / len(dataloader)
    
    if is_test:
        return avg_accuracy, avg_loss, predictions
    return avg_accuracy, avg_loss

def main():
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Load data
    train_df = load_data(f"{PATH_TO_TEST_DATA}WaM.train.tsv")
    val_df = load_data(f"{PATH_TO_TEST_DATA}WaM.dev.tsv")
    test_df = load_data(f"{PATH_TO_TEST_DATA}WaM.test.tsv", is_test=True)
    
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
    
    for epoch_i in range(NUM_EPOCHS):
        print(f'\n======== Epoch {epoch_i + 1} / {NUM_EPOCHS} ========')
        print('Training...')
        t0 = time.time()
        
        avg_train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        training_time = format_time(time.time() - t0)
        
        print(f"\nAverage training loss: {avg_train_loss:.2f}")
        print(f"Training epoch took: {training_time}")
        
        print("\nRunning Validation...")
        t0 = time.time()
        avg_val_accuracy, avg_val_loss = evaluate(model, validation_dataloader, device)
        validation_time = format_time(time.time() - t0)
        
        print(f"Validation Accuracy: {avg_val_accuracy:.2f}")
        print(f"Validation Loss: {avg_val_loss:.2f}")
        print(f"Validation took: {validation_time}")
        
        training_stats.append({
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        })
    
    print("\nTraining complete!")
    print(f"Total training took {format_time(time.time()-total_t0)}")
    
    # Display training statistics
    df_stats = pd.DataFrame(data=training_stats).set_index('epoch')
    print("\nTraining Statistics:")
    print(df_stats)
    
    # Final evaluation on test set
    print("\nRunning final evaluation on test set...")
    test_accuracy, test_loss, predictions = evaluate(model, test_dataloader, device, is_test=True)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Test Loss: {test_loss:.2f}")
    
    # Save predictions
    test_df['predicted'] = predictions
    test_df.to_csv(f"{PATH_TO_TEST_DATA}WaM.test.predictions.tsv", 
                   sep='\t', index=False)

if __name__ == "__main__":
    main()
