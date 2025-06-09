# Transformer Language Modelling

This repo deals with the implementation of a Transformer Language Model from **scratch** and visualization of intermediate attention layers which makes a word token attend to the relevant word tokens in a sentence.

1. The dataset for this project is `text8` collection which is taken from first 100M characters of Wikipedia. Only 27 characters are present (lowercase characters and spaces); special characters are replaced by a single space and numbers are spelled out as individual digits (20 becomes `two zero`).

2. For the first task, the aim is to predict, for each position in the string, how many times the character at that position occured before, maxing out at 2. For this task `text8` data is splitted into sequences of length 20. Essentially, it's 3-class classification task (with labels 0, 1, or >2 denoted as 2). This task needs to be implemented using Transformers architecture and their ability `look back` with self-attention mechanism

3. For the second task, the aim is to build a custom transformer language model. This model will consume a chunk of characters and make predictions of the next character simulataneously. For this task, we use first 100,000 characters of `text8` as the training text
