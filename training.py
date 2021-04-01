import os
import sys
import re
import unicodedata
import numpy as np
import os
import io
import logging
from downloading_data import init_logging
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
import timeit
import tqdm as tqdm
from jiwer import wer
from pick import pick
import os.path
from os import path

# Initialise logger
init_logging()
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))  # Access the parent directory
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

all_data = current_dir + "/processed_data/train/all_dialogue.txt"  # File path to all dialogue
path_to_file = current_dir + "/processed_data/train/all_training_dialogue.txt"  # File path to training dialogue


# Clean the sentences by removing punctuation and add tags at start and end of sentence
def preprocess_sentence(w):
    w = (w.lower().strip())

    # Create a space between a word and the punctuation following it
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # Replace everything with space except (a-z, A-Z, ".", "?", "!", ",", all numbers, "-" and "$" for the slotted values)
    w = re.sub(r"[^a-zA-Z?.!,¿$_0123456789-]+", " ", w)
    w = w.strip()

    w = '<start> ' + w + ' <end>'  # Start and end token added to each sentence
    return w


# Clean the sentences and create word pairs for question and response [USER, SYSTEM]
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]

    return zip(*word_pairs)


# Create a temp dataset for all the dialogue
def all_dataset(path):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    sentences = [preprocess_sentence(l) for l in lines]

    return sentences


all_lang = all_dataset(all_data)  # Temp dataset for entire dataset

# Create the tokenizer for all dialogue
lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<UNK>')
lang_tokenizer.fit_on_texts(all_lang)


# Tokenize word pairs
def tokenize(lang):
    tensor_sequence = lang_tokenizer.texts_to_sequences(lang)
    maxlen = max([len(x) for x in tensor_sequence])
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor_sequence, padding='post', truncating='post',
                                                           maxlen=maxlen)

    return tensor, maxlen


# Create input, output pairs
def load_dataset(path, num_examples=None):
    inp_lang, resp_lang = create_dataset(path, num_examples)

    input_tensor, inp_maxlen = tokenize(inp_lang)
    response_tensor, resp_maxlen = tokenize(resp_lang)

    return input_tensor, response_tensor, inp_maxlen, resp_maxlen


num_examples = 10000  # CHANGEABLE (Size of data loaded)

input_tensor, response_tensor, inp_maxlen, resp_maxlen = load_dataset(path_to_file, num_examples)

# Calculate max_length of the target tensors for future use
max_length_inp, max_length_resp = response_tensor.shape[1], input_tensor.shape[1]

# Rename training set (Already the training data)
input_tensor_train, response_tensor_train = (input_tensor, response_tensor)

print(len(input_tensor_train), len(response_tensor_train))  # Length of each set

# Initialize the tf.data dataset variables
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64  # CHANGEABLE (32 or 64)
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
embedding_dim = 256  # Can vary
units = 512  # CHANGEABLE (512 or 1024)
vocab_inp_size = len(lang_tokenizer.word_index) + 1
vocab_tar_size = len(lang_tokenizer.word_index) + 1

# Create the tf.data dateset and shuffle
dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, response_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

# Show the TensorShape size
example_input_batch, example_target_batch = next(iter(dataset))
print(example_input_batch.shape, example_target_batch.shape)


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# Sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
logger.info('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
logger.info('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

logger.info("Attention result shape: (batch size, units) {}".format(attention_result.shape))
logger.info("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)

logger.info('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([lang_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def train_model(EPOCHS):
    training_start_time = timeit.default_timer()  # Start training timer

    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # Stop training timer
    training_elapsed = timeit.default_timer() - training_start_time
    logger.info("Time taken training:", round(training_elapsed), "sec")


def evaluate(sentence):
    attention_plot = np.zeros((max_length_resp, max_length_inp))

    sentence = preprocess_sentence(sentence)

    inputs = [lang_tokenizer.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([lang_tokenizer.word_index['<start>']], 0)

    for t in range(max_length_resp):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += lang_tokenizer.index_word[predicted_id] + ' '

        if lang_tokenizer.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def response(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    # print('Input: %s' % (sentence))
    # print('Predicted response: {}'.format(result))

    return result


# Restoring the latest checkpoint in checkpoint_dir
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# ******************************************* MODEL TESTING *******************************************
# Calculates the word error rate using jiwer
def WER(gt_path, hypothesis_path):
    GTsentences = io.open(gt_path, encoding='UTF-8').read().strip().split('\n')

    Hsentences = io.open(hypothesis_path, encoding='UTF-8').read().strip().split('\n')

    error = wer(GTsentences, Hsentences)
    logger.info("Calculating the word error rate...")
    logger.info("The word error rate is: ", round((error) * 100, 2), "%", sep='')

    return error


# Main testing function
def model_testing():
    testing_start_time = timeit.default_timer()  # Start testing timer

    # Open the testing dialogue and split at new line
    with open("processed_data/test/input_testing_dialogue.txt") as f:
        content = f.readlines()
    content = [x.strip() for x in content]  # Remove \n characters

    # Pass each sentence to response function and add output to new empty list
    empty_list = []
    new_string = ''
    for x in tqdm.tqdm(content, desc='Generating responses based on testing data:'):
        result = response(x)
        new_string = ('{}'.format(result))
        new_string = new_string.replace('<end>', '')
        empty_list.append(new_string)

    # File Saving
    file_path = "processed_data/BLEU/machine_translated_dialogue.txt"
    logger.info(f"Saving Schema dialogue data to {file_path}")

    with open(file_path, mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(empty_list))

    logger.info("Saving Complete!")

    gt_path = current_dir + "/processed_data/BLEU/human_translated_dialogue.txt"
    hypothesis_path = current_dir + "/processed_data/BLEU/machine_translated_dialogue.txt"

    # Calculating the word error rate
    WER(gt_path, hypothesis_path)

    # Stop testing timer
    testing_elapsed = timeit.default_timer() - testing_start_time
    logger.info("Time taken testing:", round(testing_elapsed), "sec")


# Allow the user to choose the domain being trained
model_question = 'What action would you like to perform?  '
model_answers = ['Train model', 'Evaluate model']
model_option, index = pick(model_answers, model_question)

if index == 0:
    epoch_question = 'How many epochs to train? '
    epoch_answer = [5, 10, 25, 50, 100]
    epoch_option, index = pick(epoch_answer, epoch_question)

    if path.exists(checkpoint_dir):
        logger.info("Removing previous checkpoints...\n")
        for filename in os.listdir(checkpoint_dir):
            os.remove(checkpoint_dir + "/" + filename)

    train_model(epoch_option)

if index == 1:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    model_testing()

## BROKEN AS USER INPUTS NEED TO BE ADDED TO VOCAB LIBRARY (OPTIONAL ANYWAY)
# def user_response(sentence):
#     result, sentence, attention_plot = evaluate(sentence)
#
#     print('Input: %s' % (sentence))
#     print('Predicted response: {}'.format(result))
#
#     attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
#     plot_attention(attention_plot, sentence.split(' '), result.split(' '))
#
#     return result
#
#
# # Allow the user to input dialogue
# def user_input():
#     while True:
#         try:
#             sentence = (input("Type: "))
#         except ValueError:
#             print("Sorry, I didn't understand that.")
#             continue
#         else:
#             user_response(sentence)
#
#
# user_input()
