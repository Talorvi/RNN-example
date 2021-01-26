import string
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing


class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        skip_ids = self.ids_from_chars(['', '[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            values=[-float('inf')] * len(skip_ids),
            indices=skip_ids,
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                              return_state=True)
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        predicted_logits = predicted_logits + self.prediction_mask

        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        predicted_chars = self.chars_from_ids(predicted_ids)

        return predicted_chars, states


def generate(one_step_model, str):
    states = None
    next_char = tf.constant([str])
    result = [next_char]

    for n in range(15):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    result = result[0].numpy().decode('utf-8')
    result = result.replace(str, '')
    result = result.split()[0]
    return result.translate(str.maketrans('', '', string.punctuation))


if __name__ == "__main__":
    path_to_file = "Combined.txt"
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

    vocab = sorted(set(text))

    example_texts = ['abcdefg', 'xyz']

    chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')

    ids_from_chars = preprocessing.StringLookup(
        vocabulary=list(vocab))

    ids = ids_from_chars(chars)

    chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), invert=True)

    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 1024

    model = MyModel(
        vocab_size=len(ids_from_chars.get_vocabulary()),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units)

    model.load_weights("weights")

    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

    while True:
        input1 = input()
        print(input1)
        if input1 == "q":
            break

        results = []
        for x in range(8):
            results.append(generate(one_step_model=one_step_model, str=input1))

        results = list(dict.fromkeys(results))

        print(results)

    quit()
