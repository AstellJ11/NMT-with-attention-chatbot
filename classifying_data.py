import os
import sys
import re
import unicodedata
import numpy as np
import os
import io
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))  # Access the parent directory
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

path_to_file = current_dir + "/processed_data/train/all_training_dialog.txt"


def preprocess_sentence(w):
    w = (w.lower().strip())

    # Create a space between a word and the punctuation following it
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # Replace everything with space except (a-z, A-Z, ".", "?", "!", ",", all numbers, "-")
    w = re.sub(r"[^a-zA-Z?.!,¿0123456789-]+", " ", w)
    w = w.strip()

    w = '<start> ' + w + ' <end>'  # Start and end token added to each sentence
    return w


# Clean the sentences and create word pairs for question and response [USER, SYSTEM]
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]

    return zip(*word_pairs)


# Display the cleaned formatted word pairs as example
input, response = create_dataset(path_to_file, 50)  # Only loads the first 50 lines for efficiency of this example
print(input[2])
print(response[2])


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, lang_tokenizer


# Create input, output pairs
def load_dataset(path, num_examples=None):
    inp_lang, resp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    response_tensor, resp_lang_tokenizer = tokenize(resp_lang)

    return input_tensor, response_tensor, inp_lang_tokenizer, resp_lang_tokenizer


num_examples = 30000  # Change size of dataset here

input_tensor, response_tensor, inp_lang, resp_lang = load_dataset(path_to_file, num_examples)

# Calculate max_length of the target tensors for future use
max_length_inp, max_length_resp = response_tensor.shape[1], input_tensor.shape[1]

# Create training set # TO DO: Create testing set here as well
input_tensor_train, response_tensor_train = (input_tensor, response_tensor)

print(len(input_tensor_train), len(response_tensor_train))  # Length of each set


def convert(lang, tensor):
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang.index_word[t]))


# Display tokenized version of selected text as example
print("Input Language; index to word mapping")
convert(inp_lang, input_tensor_train[0])
print()
print("Response Language; index to word mapping")
convert(resp_lang, response_tensor_train[0])

# Initialize the tf.data dataset variables
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(resp_lang.word_index) + 1

# Create the tf.data dateset and shuffle
dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, response_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

# Show the TensorShape size
example_input_batch, example_target_batch = next(iter(dataset))
print(example_input_batch.shape, example_target_batch.shape)
