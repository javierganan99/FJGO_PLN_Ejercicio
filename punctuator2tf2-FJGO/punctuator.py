# coding: utf-8
from __future__ import division

# import models, data, main

import sys
import codecs

import tensorflow as tf
import numpy as np

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

MAX_SUBSEQUENCE_LEN = 200

print("hi!")
print(sys.version)
print(tf.config.list_physical_devices('GPU'))

# def to_array(arr, dtype=np.int32): # Convierte a numpy array
#     # minibatch of 1 sequence as column
#     return np.array([arr], dtype=dtype).T

# def convert_punctuation_to_readable(punct_token): # devuelve la puntuación a como era (sin la etiqueta)
#     if punct_token == data.SPACE:
#         return " "
#     else:
#         return punct_token[0]

# def restore(text_lines, word_vocabulary, reverse_punctuation_vocabulary, model):
#     i = 0
#     puntuated = ''

#     for text_line in text_lines:
#         if len(text_line) == 0:
#             return

#         # Si la palabra aparece en el vovabulario, se le pasa a la red, si no, se le pasa el token para caracter desconocido
#         converted_subsequence = [word_vocabulary.get(w, word_vocabulary[data.UNK]) for w in text_line]

#         # Predicción del modelo
#         y = predict(to_array(converted_subsequence), model)
#         puntuated = puntuated + (text_line[0])

#         last_eos_idx = 0
#         punctuations = []
#         for y_t in y:

#             p_i = np.argmax(tf.reshape(y_t, [-1]))
#             punctuation = reverse_punctuation_vocabulary[p_i]

#             punctuations.append(punctuation)

#             if punctuation in data.EOS_TOKENS:
#                 last_eos_idx = len(punctuations) # we intentionally want the index of next element

#         if text_line[-1] == data.END:
#             step = len(text_line) - 1
#         elif last_eos_idx != 0:
#             step = last_eos_idx
#         else:
#             step = len(text_line) - 1

#         for j in range(step):
#             puntuated = puntuated + (" " + punctuations[j] + " " if punctuations[j] != data.SPACE else " ")
#             if j < step - 1:
#                 puntuated = puntuated + (text_line[1+j])

#     return puntuated

# def predict(x, model):
#     return tf.nn.softmax(net(x))

# if __name__ == "__main__":

#     if len(sys.argv) > 1:
#         model_file = sys.argv[1]
#     else:
#         sys.exit("Model file path argument missing")

#     if len(sys.argv) > 2:
#         input_file = sys.argv[2]
#     else:
#         sys.exit("Input file path argument missing")

#     if len(sys.argv) > 3:
#         output_file = sys.argv[3]
#     else:
#         sys.exit("Output file path argument missing")

#     vocab_len = len(data.read_vocabulary(data.WORD_VOCAB_FILE))
#     x_len = vocab_len if vocab_len < data.MAX_WORD_VOCABULARY_SIZE else data.MAX_WORD_VOCABULARY_SIZE + data.MIN_WORD_COUNT_IN_VOCAB
#     x = np.ones((x_len, main.MINIBATCH_SIZE)).astype(int)

#     print("Loading model parameters...")
#     net, _ = models.load(model_file, x)

#     print("Building model...")

#     word_vocabulary = net.x_vocabulary
#     punctuation_vocabulary = net.y_vocabulary

#     reverse_word_vocabulary = {v:k for k,v in word_vocabulary.items()} # Dado un valor, me devuelve su clave (palabra)
#     reverse_punctuation_vocabulary = {v:k for k,v in punctuation_vocabulary.items()} # Dado un valor, me devuelve su clave (signo de puntuación)

#     with codecs.open(input_file, 'r', 'utf-8') as f:
#         input_text = f.readlines() # read()

#     if len(input_text) == 0:
#         sys.exit("Input file empty.")

#     text = [line.split() for line in input_text]
#     # for i,t in enumerate(text):
#     #     if t not in punctuation_vocabulary and t not in data.PUNCTUATION_MAPPING:
#     #         pass
#     #     else:
#     #         text.pop(i)

#     #print(text)
#     #text = [w for w in input_text.split() if w not in punctuation_vocabulary and w not in data.PUNCTUATION_MAPPING] + [data.END]
#     #text = [w for w in input_text.split() if w not in punctuation_vocabulary and w not in data.PUNCTUATION_MAPPING] + [data.END]
#     print(restore(text, word_vocabulary, reverse_punctuation_vocabulary, net))
