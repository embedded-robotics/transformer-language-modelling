# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, num_positions=num_positions, batched=False)
        self.trans1 = TransformerLayer(d_model, d_internal)
        self.trans2 = TransformerLayer(d_model, d_internal)
        # self.trans3 = TransformerLayer(d_model, d_internal)
        self.linear = nn.Linear(in_features=d_model, out_features=num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, indices):
        """
        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        embeddings = self.embed(indices)
        pos_embed_combined = self.pos_enc.forward(embeddings)
        out, attention1 = self.trans1(pos_embed_combined)
        out, attention2 = self.trans2(out)
        # out, attention3 = self.trans3(out)
        out = self.linear(out)
        out = self.log_softmax(out)
        return out, torch.stack([attention1, attention2])

# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        self.input_dim = d_model
        self.dff = 512
        attention_heads = 1 # if this intends to be changed, please change the forward method to inlude the corresponding attention heads
        self.dv = attention_heads * d_internal
        
        # First Attention Head
        self.weights_q1 = nn.Linear(in_features=d_model, out_features=d_internal)# query weights
        self.weights_k1 = nn.Linear(in_features=d_model, out_features=d_internal)# keys weights
        self.weights_v1 = nn.Linear(in_features=d_model, out_features=d_internal)# value weights
        self.softmax1 = nn.Softmax(dim=1)# softmax over attention

        # Second Attention Head
        self.weights_q2 = nn.Linear(in_features=d_model, out_features=d_internal)# query weights
        self.weights_k2 = nn.Linear(in_features=d_model, out_features=d_internal)# keys weights
        self.weights_v2 = nn.Linear(in_features=d_model, out_features=d_internal)# value weights
        self.softmax2 = nn.Softmax(dim=1)# softmax over attention

        # Third Attention Head
        self.weights_q3 = nn.Linear(in_features=d_model, out_features=d_internal)# query weights
        self.weights_k3 = nn.Linear(in_features=d_model, out_features=d_internal)# keys weights
        self.weights_v3 = nn.Linear(in_features=d_model, out_features=d_internal)# value weights
        self.softmax3 = nn.Softmax(dim=1)# softmax over attention

        # Fourth Attention Head
        self.weights_q4 = nn.Linear(in_features=d_model, out_features=d_internal)# query weights
        self.weights_k4 = nn.Linear(in_features=d_model, out_features=d_internal)# keys weights
        self.weights_v4 = nn.Linear(in_features=d_model, out_features=d_internal)# value weights
        self.softmax4 = nn.Softmax(dim=1)# softmax over attention
        
        # Multihead combination
        self.weights_out = nn.Linear(in_features=self.dv, out_features=d_model)# o1 weights
        # feedforward layer
        self.ffn_linear1 = nn.Linear(in_features=d_model, out_features=self.dff)
        # non-linear layer
        self.ffn_nonlinear = nn.ReLU()
        # output linear layer
        self.ffn_linear2 = nn.Linear(in_features=self.dff, out_features=d_model)

    def forward(self, input_vecs):

        # First Attention Head
        query_mat1 = self.weights_q1(input_vecs)
        key_mat1 = self.weights_k1(input_vecs)
        value_mat1 = self.weights_v1(input_vecs)
        attention1 = torch.matmul(query_mat1, key_mat1.T)
        attention_normalized1 = self.softmax1(attention1/(torch.sqrt(torch.tensor([self.input_dim]))))
        value_out1 = torch.matmul(attention_normalized1, value_mat1)

        # Second Attention Head
        query_mat2 = self.weights_q2(input_vecs)
        key_mat2 = self.weights_k2(input_vecs)
        value_mat2 = self.weights_v2(input_vecs)
        attention2 = torch.matmul(query_mat2, key_mat2.T)
        attention_normalized2 = self.softmax1(attention2/(torch.sqrt(torch.tensor([self.input_dim]))))
        value_out2 = torch.matmul(attention_normalized2, value_mat2)

        # Third Attention Head
        query_mat3 = self.weights_q3(input_vecs)
        key_mat3 = self.weights_k3(input_vecs)
        value_mat3 = self.weights_v3(input_vecs)
        attention3 = torch.matmul(query_mat3, key_mat3.T)
        attention_normalized3 = self.softmax1(attention3/(torch.sqrt(torch.tensor([self.input_dim]))))
        value_out3 = torch.matmul(attention_normalized3, value_mat3)

        # Fourth Attention Head
        query_mat4 = self.weights_q4(input_vecs)
        key_mat4 = self.weights_k4(input_vecs)
        value_mat4 = self.weights_v4(input_vecs)
        attention4 = torch.matmul(query_mat4, key_mat4.T)
        attention_normalized4 = self.softmax1(attention4/(torch.sqrt(torch.tensor([self.input_dim]))))
        value_out4 = torch.matmul(attention_normalized4, value_mat4)

        # Multihead attention combine
        multi_head_out = self.weights_out(value_out1)
        # multi_head_out = self.weights_out(torch.cat([value_out1, value_out2, value_out3, value_out4], dim=1))

        # Residual after attention head
        residual1 = input_vecs + multi_head_out
        # Feed forward layer
        ffn_lin1 = self.ffn_linear1(residual1)
        ffn_nonlin1 = self.ffn_nonlinear(ffn_lin1)
        ffn_lin2 = self.ffn_linear2(ffn_nonlin1)
        # Residual after feed forward layer
        residual2 = residual1 + ffn_lin2
        return residual2, attention_normalized1

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

# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    
    vocab_size = 27
    d_model = 128
    d_internal = 100
    num_positions = 20
    num_classes = 3
    num_layers = None
    num_epochs = 5

    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)
        loss_fcn = nn.NLLLoss()
        for ex_idx in ex_idxs:
            log_probs, _ = model.forward(train[ex_idx].input_tensor)
            loss = loss_fcn(log_probs, train[ex_idx].output_tensor)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        print(f"Epoch: {t+1}, Total Loss: {loss_this_epoch}")
    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False, do_attention_normalization_test=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
        do_attention_normalization_test = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        if do_attention_normalization_test:
            normalizes = attention_normalization_test(attn_maps)
            print("%s normalization test on attention maps" % ("Passed" if normalizes else "Failed"))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))


def attention_normalization_test(attn_maps):
    """
    Tests that the attention maps sum to one over rows
    :param attn_maps: the list of attention maps
    :return:
    """
    for attn_map in attn_maps:
        total_prob_over_rows = torch.sum(attn_map, dim=1)
        if torch.any(total_prob_over_rows < 0.99).item() or torch.any(total_prob_over_rows > 1.01).item():
            print("Failed normalization test: probabilities not sum to 1.0 over rows")
            print("Total probability over rows:", total_prob_over_rows)
            return False
    return True
