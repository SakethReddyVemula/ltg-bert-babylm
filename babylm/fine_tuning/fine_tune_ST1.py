# Import models
import torch
import os
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from model import BertForSequenceClassification
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import random
from tokenizers import Tokenizer

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device: ", device)

df = pd.read_csv("../data/WaD.train.tsv", delimiter = '\t', header = None, names = ['word_a', 'word_b', 'label'], skiprows = 1)

# Load training Data
df = pd.read_csv("../data/WaD.train.tsv", delimiter = '\t', header = None, names = ['word_a', 'word_b', 'label'], skiprows = 1)
df.dropna(inplace=True)
print(f"Number of Training instances: {df.shape[0]}")

# Load validation Data
val_df = pd.read_csv("../data/WaD.dev.tsv", header=None, delimiter='\t', skiprows=1, names=['word_a', 'word_b', 'label'])
val_df.dropna(inplace=True)
print(f"Number of Validation instances: {val_df.shape[0]}")

words_a = df.word_a.values
words_b = df.word_b.values
labels = df.label.values
val_words_a = val_df.word_a.values
val_words_b = val_df.word_b.values
val_labels = val_df.label.values

# Load pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("../Tokenizer/", do_lower_case = True)
# tokenizer = Tokenizer.from_file("../tokenizer.json")

print("word_a: ", words_a[0])
print("Tokenized word_a: ", tokenizer.tokenize(words_a[0]))
print("Tokenized ids word_a: ", tokenizer.convert_tokens_to_ids(tokenizer.tokenize(words_a[0])))
print("word_b: ", words_b[0])
print("Tokenized word_b: ", tokenizer.tokenize(words_b[0]))
print("Tokenized ids word_b: ", tokenizer.convert_tokens_to_ids(tokenizer.tokenize(words_b[0])))

max_len = 0

for sent in zip(words_a, words_b):
    input_ids = tokenizer.encode(sent[0], sent[1], add_special_tokens = True)
    max_len = max(max_len, len(input_ids))

print("Max sentence length: ", max_len)

input_ids = []
attention_masks = []

for sent in zip(words_a, words_b):
    encoded_dict = tokenizer.encode_plus(sent[0],
                                        sent[1],
                                        add_special_tokens = True,
                                        max_length = 128,
                                        padding = 'max_length',
                                        return_attention_mask = True,
                                        return_tensors = 'pt',)
    
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    
input_ids = torch.cat(input_ids, dim = 0)
attention_masks = torch.cat(attention_masks, dim = 0)
labels = torch.tensor(labels)

print("word_a: ", words_a[0])
print("word_b: ", words_b[0])
print("token IDs: ", input_ids[0])
print("Attention Mask: ", attention_masks[0])

val_input_ids = []
val_attention_masks = []

for sent in zip(val_words_a, val_words_b):
    encoded_dict = tokenizer.encode_plus(sent[0],
                                        sent[1],
                                        add_special_tokens = True,
                                        max_length = 128,
                                        padding = 'max_length',
                                        return_attention_mask = True,
                                        return_tensors = 'pt',)
    
    val_input_ids.append(encoded_dict['input_ids'])
    val_attention_masks.append(encoded_dict['attention_mask'])
    
val_input_ids = torch.cat(val_input_ids, dim = 0)
val_attention_masks = torch.cat(val_attention_masks, dim = 0)
val_labels = torch.tensor(val_labels)

print("word_a: ", val_words_a[0])
print("word_b: ", val_words_b[0])
print("token IDs: ", val_input_ids[0])
print("Attention Mask: ", val_attention_masks[0])

train_dataset = TensorDataset(input_ids, attention_masks, labels)
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
train_size = len(train_dataset)
val_size = len(val_dataset)
print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

batch_size = 64

train_dataloader = DataLoader(train_dataset,
                             sampler = RandomSampler(train_dataset),
                             batch_size = batch_size)

validation_dataloader = DataLoader(val_dataset,
                           sampler = RandomSampler(val_dataset),
                           batch_size = batch_size)

model = BertForSequenceClassification(
    config_path="../configs/small.json",
    vocab_path="../tokenizer.json",
    num_labels=2,
    model_path="../checkpoints/small/pytorch_model.bin"
)


model.to(device)

# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

optimizer = optim.AdamW(model.parameters(), lr = 1e-4, eps = 1e-8, weight_decay=0.1)

epochs = 10

total_steps = len(train_dataloader) * epochs
print("Total Steps: ", total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                          num_warmup_steps = 0,
                                          num_training_steps = total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)/len(pred_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

training_stats = []
total_t0 = time.time()

for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    t0 = time.time()
    total_train_loss = 0
    model.train()
    
    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
        # if 1:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        model.zero_grad()
        
        result = model(b_input_ids,
                      attention_mask = b_input_mask,
                      labels = b_labels,
                      return_dict = True)
        # print(result)

        # loss = result.loss
        loss = result['loss']
        # logits = result.logits
        logits = result['logits']
        
        total_train_loss += loss.item()
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
    avg_train_loss = total_train_loss / len(train_dataloader)
    
    training_time = format_time(time.time() - t0)
    
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
    
    print("")
    print("Running Validation...")
    
    t0 = time.time()
    
    model.eval()
    
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():
            result = model(b_input_ids,
                          attention_mask = b_input_mask,
                          labels = b_labels,
                          return_dict = True)
        
        # loss = result.loss
        loss = result['loss']
        # logits = result.logits
        logits = result['logits']

        total_eval_loss += loss.item()
        
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))
    
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )
    
print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

import pandas as pd

# Display floats with two decimal places.
pd.set_option('display.precision', 2)

# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)

# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')

# A hack to force the column headers to wrap.
# df_stats = df_stats.style.set_table_styles([dict(selector="th", props=[('max-width', '70px')])])

# Display the table.
print(df_stats)

import pandas as pd

df = pd.read_csv("../data/WaD.test.tsv", delimiter='\t', header = None, names = ['word_a', 'word_b', 'label'], skiprows = 1)
# df.dropna(inplace = True)

print("Number of test sentences: {:,}\n".format(df.shape[0]))
# df.sample(10)

test_words_a = df.word_a.values
test_words_b = df.word_b.values
val_labels = df.label.values

input_ids = []
attention_masks = []

for sent in zip(test_words_a, test_words_b):
    encoded_dict = tokenizer.encode_plus(sent[0],
                                        sent[1],
                                        add_special_tokens=True,
                                        max_length = 128,
                                        padding = 'max_length',
                                        return_attention_mask = True,
                                        return_tensors = 'pt',
                                        )
    
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim = 0)
attention_masks = torch.cat(attention_masks, dim = 0)
val_labels = torch.tensor(val_labels)

batch_size = 64

prediction_data = TensorDataset(input_ids, attention_masks, val_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler = prediction_sampler, batch_size = batch_size)
print(f"Length of prediciton_dataloader: {len(prediction_dataloader)}")

model.eval()

predictions, true_labels = [], []

for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    
    with torch.no_grad():
        result = model(b_input_ids,
                      attention_mask = b_input_mask,
                      return_dict = True)
    
    # logits = result.logits
    logits = result['logits']
    
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
#     print(predictions)
    predictions.append(logits)
    true_labels.append(label_ids)
    
print("Done evaluating!")

print(f'Positive samples: {df.label.sum()} of {len(df.label)} ({df.label.sum() / len(df.label) * 100})%')



