# lm.py

import argparse
import json
import time
from transformer_lm import *
from utils import *

####################################################
# DO NOT MODIFY THIS FILE IN YOUR FINAL SUBMISSION #
####################################################


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='lm.py')
    parser.add_argument('--model', type=str, default='UNIFORM', help='model to run (UNIFORM or NEURAL)')
    parser.add_argument('--train_path', type=str, default='data/text8-100k.txt', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/text8-dev.txt', help='path to dev set (you should not need to modify)')
    parser.add_argument('--output_bundle_path', type=str, default='output.json', help='path to write the results json to (you should not need to modify)')
    args = parser.parse_args()
    return args


def read_text(file):
    """
    :param file:
    :return: The text in the given file as a single string
    """
    all_text = ""
    for line in open(file):
        all_text += line
    print("%i chars read in" % len(all_text))
    return all_text


def run_sanity_check(lm, vocab_index):
    """
    Runs two sanity checks: (1) The language model must return valid probabilities for a few contexts. This checks that
    your model can take sequences of different lengths and contexts of different lengths without crashing.
    :param lm: the trained LM
    :return: True if the output is sane, false otherwise
    """
    contexts = [" ", " a person ", " some person "]
    next_seqs = ["s", "sits", "stands"]
    for context in contexts:
        for next_seq in next_seqs:
            for i in range(0, len(next_seq)):
                # print(repr(context + next_seq[0:i]))
                # print(repr(next_seq[i]))
                next_char_log_probs = lm.get_next_char_log_probs(context + next_seq[0:i])
                # print(repr(next_char_log_probs))
                single_log_prob = next_char_log_probs[vocab_index.index_of(next_seq[i])]
                if single_log_prob > 0.0:
                    print("ERROR: sanity checks failed, next char log probability %f is invalid" % (single_log_prob))
                    return False
    return True


def normalization_test(lm, vocab_index):
    """
    Tests that LM normalizes, checks multiple contexts and sums over everything in the vocabulary to make sure it
    sums to one
    :param lm:
    :param voc:
    :return:
    """
    contexts = [" ", " a person "]
    normalizes = True
    for context in contexts:
        total_prob_single = np.sum(np.exp(lm.get_next_char_log_probs(context)))
        if total_prob_single < 0.99 or total_prob_single > 1.01:
            normalizes = False
            print("Probabilities sum to %f not 1.0 for context %s" % (total_prob_single, context))
    return normalizes

def perplexity_range_check(perplexity):
    if perplexity < 3.5:
        print("ERROR: checks failed, the perplexity is too low")
        return False
    return True

def print_evaluation(text, lm, vocab_index, output_bundle_path):
    """
    Runs both the sanity check and also runs the language model on the given text and prints three metrics: log
    probability of the text under this model (treating the text as one log sequence), average log probability (the
    previous value divided by sequence length), and perplexity (averaged "branching favor" of the model)
    :param text: the text to evaluate
    :param lm: model to evaluate
    :param output_bundle_path: the path to print the output bundle to, in addition to printing it
    """
    sane = run_sanity_check(lm, vocab_index)
    log_prob = 0.0
    for i in range(0, len(text)):
        next_char_log_probs = lm.get_next_char_log_probs(text[max(0, i-19):i])
        log_prob += next_char_log_probs[vocab_index.index_of(text[i])]    
    avg_log_prob = log_prob/len(text)
    perplexity = np.exp(-log_prob / len(text))
    normalizes = normalization_test(lm, vocab_index)
    range_check = perplexity_range_check(perplexity)
    data = {'sane': sane, 'normalizes': normalizes, 'range': range_check, 'log_prob': log_prob, 'avg_log_prob': avg_log_prob, 'perplexity': perplexity}
    print("=====Results=====")
    print(json.dumps(data, indent=2))
    with open(output_bundle_path, 'w') as outfile:
        json.dump(data, outfile)
    return data


if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()
    print(args)

    train_text = read_text(args.train_path)
    dev_text = read_text(args.dev_path)

    # Vocabs is lowercase letters a to z and space
    vocab = [chr(ord('a') + i) for i in range(0, 26)] + [' ']
    vocab_index = Indexer()
    for char in vocab:
        vocab_index.add_and_get_index(char)
    print(repr(vocab_index))

    print("First 100 characters of train:")
    print(train_text[0:100])
    # Train our model
    if args.model == "NEURAL":
        model = train_lm(args, train_text, dev_text, vocab_index)
    elif args.model == "UNIFORM":
        model = UniformLanguageModel(len(vocab))
    else:
        raise Exception("Pass in either UNIFORM or NEURAL to run the appropriate system")

    print_evaluation(dev_text, model, vocab_index, args.output_bundle_path)
