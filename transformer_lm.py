# models.py

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import random

# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param d_model: embeddings dimensions of the model
        :param num_classes: number of classes predicted at the output layer; should be 27 since we have 27 total characters
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)
        self.linear1 = nn.Linear(in_features=d_model, out_features=num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, indices):
        """
        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a chunk_lenx27 matrix)
        """
        embeddings = self.embed(indices)
        pos_embed_combined = self.pos_enc.forward(embeddings)
        mask = torch.triu(torch.ones(len(indices), len(indices)) * float('-inf'), diagonal=1)
        trans_out = self.transformer_encoder(pos_embed_combined, mask=mask)
        lin_out = self.linear1(trans_out)
        log_probs = self.log_softmax(lin_out)
        return log_probs

class NeuralLanguageModel(LanguageModel):
    def __init__(self, transformer_lm, vocab_index):
        self.transformer_lm = transformer_lm
        self.vocab_index = vocab_index
    
    def get_next_char_log_probs(self, context):
        
        if len(context) == 0:
            arr = np.array([self.vocab_index.index_of(' ')])
        else:
            arr = np.array([self.vocab_index.index_of(i) for i in context])

        arr_tensor = torch.LongTensor(arr)
        
        log_probs = self.transformer_lm.forward(arr_tensor)
        log_probs = log_probs.cpu().detach().numpy()
        return log_probs[-1] # only return the log probabilities of the last element since this is the next element

# Takes train text, then forms a list of input and output indices of a specific chunk length
# Input indices start with space followed by (chunk_len-1)
# Output indices are the indices of the actual chunk_len
def get_input_output_indices(text, vocab_index, chunk_len):
    text_len = len(text)

    text_ranges = np.arange(0, text_len, chunk_len)
    text_ranges = np.append(text_ranges, text_len)

    input_bundles = []
    output_bundles = []

    for i in range(0, len(text_ranges)-1):
        output_data = text[text_ranges[i]:text_ranges[i+1]]
        input_data = ' ' + output_data[0:chunk_len-1]
        output_indexed = np.array([vocab_index.index_of(i) for i in output_data])
        input_indexed = np.array([vocab_index.index_of(i) for i in input_data])
        ouptut_tensor = torch.LongTensor(output_indexed)
        input_tensor = torch.LongTensor(input_indexed)
        output_bundles.append(ouptut_tensor)
        input_bundles.append(input_tensor)
    
    return input_bundles, output_bundles


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    
    vocab_size = len(vocab_index)
    d_model = 128
    chunk_len = 20
    num_classes = vocab_size
    num_layers = 2
    num_epochs = 10
    
    # Get input and output lists of incdices for training text
    train_input_indices, train_output_indices = get_input_output_indices(train_text, vocab_index, chunk_len)

    # Get input and output lists of incdices for training text
    dev_input_indices, dev_output_indices = get_input_output_indices(dev_text, vocab_index, chunk_len)
        
    # Defining the model and parameters
    language_model = TransformerLanguageModel(vocab_size, d_model, num_classes, num_layers)
    language_model.zero_grad()
    language_model.train()
    optimizer = optim.Adam(language_model.parameters(), lr=1e-4)
    loss_fcn = nn.NLLLoss()
    
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        for i in range(0, len(train_input_indices)):
            log_probs = language_model.forward(train_input_indices[i])
            loss = loss_fcn(log_probs, train_output_indices[i])
            language_model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        print(f"Epoch: {t+1}, Total Loss: {loss_this_epoch}")
    language_model.eval()
    
    neural_language_model = NeuralLanguageModel(language_model, vocab_index)
    return neural_language_model