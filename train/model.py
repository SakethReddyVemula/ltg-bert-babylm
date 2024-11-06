import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _softmax_backward_data as _softmax_backward_data
from torch.utils import checkpoint
from typing import Optional, List
from typing import Union, Tuple
from tokenizers import Tokenizer
from config import BertConfig


class Bert(nn.Module):
    def __init__(self, config, activation_checkpointing=False):
        super().__init__()
        self.embedding = Embedding(config)
        self.transformer = Encoder(config, activation_checkpointing)
        self.classifier = MaskClassifier(config, self.embedding.word_embedding.weight)

    def get_contextualized(self, input_ids, attention_mask):
        static_embeddings, relative_embedding = self.embedding(input_ids)
        contextualized_embeddings = self.transformer(static_embeddings, attention_mask.unsqueeze(1).unsqueeze(2), relative_embedding)
        return contextualized_embeddings
    
    def get_contextualized_embeddings(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("You have to specify input_ids")

        batch_size, seq_length = input_shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
        else:
            attention_mask = ~attention_mask.bool()
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
 
        static_embeddings, relative_embedding = self.embedding(input_ids.t())
        contextualized_embeddings, attention_probs = self.transformer.forward_all_hidden_states(static_embeddings, attention_mask, relative_embedding)
        contextualized_embeddings = [e.transpose(0, 1) for e in contextualized_embeddings]
        last_layer = contextualized_embeddings[-1]
        contextualized_embeddings = [contextualized_embeddings[0]] + [
            contextualized_embeddings[i] - contextualized_embeddings[i - 1]
            for i in range(1, len(contextualized_embeddings))
        ]
        return last_layer, contextualized_embeddings, attention_probs
    
    @torch.no_grad()
    def get_attention_scores(self, input_ids, attention_mask):
        static_embeddings, relative_embedding = self.embedding(input_ids)
        attention_scores = self.transformer.forward_attention_scores(static_embeddings, attention_mask.unsqueeze(1).unsqueeze(2), relative_embedding)
        return attention_scores
    
    @torch.no_grad()
    def get_second_to_last_hidden_states(self, input_ids, attention_mask):
        static_embeddings, relative_embedding = self.embedding(input_ids)
        hidden_states = self.transformer.forward_second_to_last_hidden_states(static_embeddings, attention_mask.unsqueeze(1).unsqueeze(2), relative_embedding)
        return hidden_states

    def forward(self, input_ids, attention_mask, masked_lm_labels=None):
        contextualized_embeddings = self.get_contextualized(input_ids, attention_mask)
        subword_prediction = self.classifier(contextualized_embeddings, masked_lm_labels)
        return subword_prediction


class Encoder(nn.Module):
    def __init__(self, config, activation_checkpointing=False):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

        for i, layer in enumerate(self.layers):
            layer.mlp.mlp[1].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))
            layer.mlp.mlp[-2].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))

        self.activation_checkpointing = activation_checkpointing

    def forward(self, hidden_states, attention_mask, relative_embedding):
        for layer in self.layers:
            if self.activation_checkpointing:
                hidden_states, _ = checkpoint.checkpoint(layer, hidden_states, attention_mask, relative_embedding)
            else:
                hidden_states, _ = layer(hidden_states, attention_mask, relative_embedding)
        return hidden_states
    
    def forward_attention_scores(self, hidden_states, attention_mask, relative_embedding):
        attention_scores = torch.zeros(hidden_states.size(1), hidden_states.size(0), hidden_states.size(0), device=hidden_states.device, dtype=torch.float32)  # shape: [batch_size, seq_len, seq_len]

        for layer in self.layers:
            if self.activation_checkpointing:
                hidden_states, attention_score = checkpoint.checkpoint(layer, hidden_states, attention_mask, relative_embedding)
            else:
                hidden_states, attention_score = layer(hidden_states, attention_mask, relative_embedding)
            attention_scores += attention_score

        attention_scores /= len(self.layers)
        return attention_scores

    def forward_all_hidden_states(self, hidden_states, attention_mask, relative_embedding):
        hidden_states = [hidden_states]
        attention_scores = torch.zeros(hidden_states[0].size(1), hidden_states[0].size(0), hidden_states[0].size(0), device=hidden_states[0].device, dtype=torch.float32)  # shape: [batch_size, seq_len, seq_len]

        for layer in self.layers:
            if self.activation_checkpointing:
                hidden_state, attention_score = checkpoint.checkpoint(layer, hidden_states[-1], attention_mask, relative_embedding)
            else:
                hidden_state, attention_score = layer(hidden_states[-1], attention_mask, relative_embedding)
            hidden_states.append(hidden_state)
            attention_scores += attention_score

        attention_scores /= len(self.layers)
        return hidden_states, attention_scores
    
    def forward_second_to_last_hidden_states(self, hidden_states, attention_mask, relative_embedding):
        for layer in self.layers[:-1]:
            if self.activation_checkpointing:
                hidden_states, _ = checkpoint.checkpoint(layer, hidden_states, attention_mask, relative_embedding)
            else:
                hidden_states, _ = layer(hidden_states, attention_mask, relative_embedding)
        return hidden_states


class MaskClassifier(nn.Module):
    def __init__(self, config, subword_embedding):
        super().__init__()
        self.nonlinearity = nn.Sequential(
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(subword_embedding.size(1), subword_embedding.size(0))
        )
        self.initialize(config.hidden_size, subword_embedding)

    def initialize(self, hidden_size, embedding):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.nonlinearity[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.nonlinearity[-1].weight = embedding
        self.nonlinearity[1].bias.data.zero_()
        self.nonlinearity[-1].bias.data.zero_()

    def forward(self, x, masked_lm_labels=None):
        if masked_lm_labels is not None:
            x = torch.index_select(x.flatten(0, 1), 0, torch.nonzero(masked_lm_labels.flatten() != -100).squeeze())
        x = self.nonlinearity(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.mlp = FeedForward(config)

    def forward(self, x, padding_mask, relative_embedding):
        x_, attention_scores = self.attention(x, padding_mask, relative_embedding)
        x = x + x_
        x = x + self.mlp(x)
        return x, attention_scores


class GeGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        x = x * F.gelu(gate, approximate='tanh')
        return x


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, 2*config.intermediate_size, bias=False),
            GeGLU(),
            nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.initialize(config.hidden_size)

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.mlp[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.mlp[-2].weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, x):
        return self.mlp(x)


class MaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(self, x, mask, dim):
        # print(x.shape)
        # print(mask.shape)
        # print(x)
        # print(mask)
        self.dim = dim
        x.masked_fill_(mask.to(bool), float('-inf'))
        x = torch.softmax(x, self.dim)
        x.masked_fill_(mask.to(bool), 0.0)
        self.save_for_backward(x)
        return x

    @staticmethod
    def backward(self, grad_output):
        output, = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output.dtype)
        return inputGrad, None, None


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"The hidden size {config.hidden_size} is not a multiple of the number of attention heads {config.num_attention_heads}")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads

        self.in_proj_qk = nn.Linear(config.hidden_size, 2*config.hidden_size, bias=True)
        self.in_proj_v = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.pre_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False)
        self.post_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)

        position_indices = torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(1) \
            - torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(0)
        position_indices = self.make_log_bucket_position(position_indices, config.position_bucket_size, config.max_position_embeddings)
        position_indices = config.position_bucket_size - 1 + position_indices
        self.register_buffer("position_indices", position_indices, persistent=True)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale = 1.0 / math.sqrt(3 * self.head_size)
        self.initialize()

    def make_log_bucket_position(self, relative_pos, bucket_size, max_position):
        sign = torch.sign(relative_pos)
        mid = bucket_size // 2
        abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, torch.abs(relative_pos).clamp(max=max_position - 1))
        log_pos = torch.ceil(torch.log(abs_pos / mid) / math.log((max_position-1) / mid) * (mid - 1)).int() + mid
        bucket_pos = torch.where(abs_pos <= mid, relative_pos, log_pos * sign).long()
        return bucket_pos

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.in_proj_qk.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.in_proj_v.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.in_proj_qk.bias.data.zero_()
        self.in_proj_v.bias.data.zero_()
        self.out_proj.bias.data.zero_()

    def forward(self, hidden_states, attention_mask, relative_embedding):
        key_len, batch_size, _ = hidden_states.size()
        query_len = key_len

        if self.position_indices.size(0) < query_len:
            position_indices = torch.arange(query_len, dtype=torch.long).unsqueeze(1) \
                - torch.arange(query_len, dtype=torch.long).unsqueeze(0)
            position_indices = self.make_log_bucket_position(position_indices, self.config.position_bucket_size, 512)
            position_indices = self.config.position_bucket_size - 1 + position_indices
            self.register_buffer("position_indices", position_indices.to(hidden_states.device), persistent=True)

        hidden_states = self.pre_layer_norm(hidden_states)

        query, key = self.in_proj_qk(hidden_states).chunk(2, dim=2)  # shape: [T, B, D]
        value = self.in_proj_v(hidden_states)  # shape: [T, B, D]

        pos = self.in_proj_qk(self.dropout(relative_embedding))  # shape: [2T-1, 2D]
        pos = F.embedding(self.position_indices[:query_len, :key_len], pos)  # shape: [T, T, 2D]
        query_pos, key_pos = pos.chunk(2, dim=-1)
        query_pos = query_pos.view(query_len, key_len, self.num_heads, self.head_size)
        key_pos = key_pos.view(query_len, key_len, self.num_heads, self.head_size)

        query = query.reshape(query_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        key = key.reshape(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        value = value.view(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)

        attention_scores = torch.bmm(query, key.transpose(1, 2) * self.scale)

        query = query.view(batch_size, self.num_heads, query_len, self.head_size)
        key = key.view(batch_size, self.num_heads, query_len, self.head_size)
        attention_scores = attention_scores.view(batch_size, self.num_heads, query_len, key_len)
        attention_scores.add_(torch.einsum("bhqd,qkhd->bhqk", query, key_pos * self.scale))
        attention_scores.add_(torch.einsum("bhkd,qkhd->bhqk", key * self.scale, query_pos))

        returned_attention_scores = attention_scores.detach().clone().mean(dim=1)

        attention_probs = MaskedSoftmax.apply(attention_scores, attention_mask, -1)

        attention_probs = self.dropout(attention_probs)
        context = torch.bmm(attention_probs.flatten(0, 1), value)  # shape: [B*H, Q, D]
        context = context.transpose(0, 1).reshape(context.size(1), -1, self.hidden_size)  # shape: [Q, B, H*D]
        context = self.out_proj(context)
        context = self.post_layer_norm(context)
        context = self.dropout(context)

        return context, returned_attention_scores


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.relative_embedding = nn.Parameter(torch.empty(2 * config.position_bucket_size - 1, config.hidden_size))
        self.relative_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.initialize()

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.relative_embedding, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.word_embedding.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, input_ids):
        word_embedding = self.dropout(self.word_layer_norm(self.word_embedding(input_ids)))
        relative_embeddings = self.relative_layer_norm(self.relative_embedding)
        return word_embedding, relative_embeddings
    
"""
    Hugging Face Wrappers
"""
class BertForMaskedLM:
    def __init__(self, model_path, config_path, vocab_path):
        self.config = BertConfig(config_path)
        self.model = Bert(self.config)
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        
        self.model.eval()
        
        self.tokenizer = Tokenizer.from_file(vocab_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.cls_index = self.tokenizer.token_to_id("[CLS]")
        self.sep_index = self.tokenizer.token_to_id("[SEP]")
        self.pad_index = self.tokenizer.token_to_id("[PAD]")
        self.mask_index = self.tokenizer.token_to_id("[MASK]")

    def prepare_input(self, text):
        encoded = self.tokenizer.encode(text)
        tokens = [self.tokenizer.token_to_id(token) for token in encoded.tokens]
        
        seq_length = 128
        target_seq_length = seq_length - 2
        tokens = tokens[:target_seq_length]
        
        padding_length = target_seq_length - len(tokens)
        segment = [self.cls_index] + tokens + [self.sep_index] + [self.pad_index] * padding_length
        
        input_ids = torch.LongTensor(segment).unsqueeze(0).to(self.device)
        
        attention_mask = torch.cat([
            torch.zeros(len(tokens) + 2, dtype=torch.bool),
            torch.ones(padding_length, dtype=torch.bool)
        ]).unsqueeze(0).to(self.device)
        print(f"Input_ids shape: {input_ids.shape}")
        print(f"Attention_mask shape: {attention_mask.shape}")
        
        return input_ids, attention_mask

    def predict_masked_words(self, text, top_k=20):
        input_ids, attention_mask = self.prepare_input(text)
        
        masked_indices = (input_ids == self.mask_index).nonzero(as_tuple=True)[1]
        
        with torch.no_grad():
            outputs = self.model(input_ids.t(), attention_mask)  
            print(f"outputs shape: {outputs.shape}") # outputs shape: [128, 1, 16384]

        predictions = []
        for idx in masked_indices:
            # Extract logits for the masked position
            logits = outputs[idx, 0, :]  # Shape: [16384]
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            
            top_k_tokens = [self.tokenizer.id_to_token(idx.item()) for idx in top_k_indices]
            predictions.append(list(zip(top_k_tokens, top_k_probs.tolist())))
        
        return predictions
    
    
class Classifier(nn.Module):
    def __init__(self, config, num_labels: int):
        super().__init__()

        drop_out = getattr(config, "classifier_dropout", config.hidden_dropout_prob)

        self.nonlinearity = nn.Sequential(
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Dropout(drop_out),
            nn.Linear(config.hidden_size, num_labels)
        )
        self.initialize(config.hidden_size)

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.nonlinearity[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.nonlinearity[-1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.nonlinearity[1].bias.data.zero_()
        self.nonlinearity[-1].bias.data.zero_()

    def forward(self, x):
        x = self.nonlinearity(x)
        return x
    

class BertForSequenceClassification(Bert):
    _keys_to_ignore_on_load_unexpected = ["classifier"]
    _keys_to_ignore_on_load_missing = ["head"]

    def __init__(self, config_path, vocab_path, num_labels, model_path=None):
        self.config = BertConfig(config_path)
        super().__init__(self.config)
        self.bert = Bert(self.config)
        checkpoint = torch.load(model_path, map_location='cpu')
        self.bert.load_state_dict(checkpoint['model'])
        
        self.bert.eval()
        
        self.tokenizer = Tokenizer.from_file(vocab_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert.to(self.device)
        self.num_labels = num_labels

        self.cls_index = self.tokenizer.token_to_id("[CLS]")
        self.sep_index = self.tokenizer.token_to_id("[SEP]")
        self.pad_index = self.tokenizer.token_to_id("[PAD]")
        self.mask_index = self.tokenizer.token_to_id("[MASK]")
        

        # Add a classification head
        self.head = Classifier(self.config, self.num_labels)

        # Load pre-trained weights if provided
        if model_path:
            self.load_pretrained(model_path)

    def load_pretrained(self, model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        self.bert.load_state_dict(checkpoint['model'])
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], dict]:
        
        return_dict = return_dict if return_dict is not None else True
        sequence_output, contextualized_embeddings, attention_probs = self.get_contextualized_embeddings(input_ids, attention_mask)
        logits = self.head(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + output[2:]
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': output if output_hidden_states else None,
            'attentions': None  # Bert model doesn't seem to return attention probs
        }
    
    def freeze_bert_encoder(self):
        """Freeze all parameters in the BERT encoder"""
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        """Unfreeze all parameters in the BERT encoder"""
        for param in self.bert.parameters():
            param.requires_grad = True

    def prepare_input(self, text):
        encoded = self.tokenizer.encode(text)
        tokens = [self.tokenizer.token_to_id(token) for token in encoded.tokens]
        
        seq_length = 128
        target_seq_length = seq_length - 2
        tokens = tokens[:target_seq_length]
        
        padding_length = target_seq_length - len(tokens)
        segment = [self.cls_index] + tokens + [self.sep_index] + [self.pad_index] * padding_length
        
        input_ids = torch.LongTensor(segment).unsqueeze(0).to(self.device)
        
        attention_mask = torch.cat([
            torch.zeros(len(tokens) + 2, dtype=torch.bool),
            torch.ones(padding_length, dtype=torch.bool)
        ]).unsqueeze(0).to(self.device)
        return input_ids, attention_mask
    

