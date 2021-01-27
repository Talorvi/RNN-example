import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import os
from tensorflow.python.keras.callbacks import ModelCheckpoint

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

chars = chars_from_ids(ids)

tf.strings.reduce_join(chars, axis=-1).numpy()


def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))

ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)

sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


split_input_target(list("Tensorflow"))

dataset = sequences.map(split_input_target)

BATCH_SIZE = 128
BUFFER_SIZE = 10000

dataset = (
    dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024


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


model = MyModel(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

example_batch_loss = loss(target_example_batch, example_batch_predictions)
mean_loss = example_batch_loss.numpy().mean()

tf.exp(mean_loss).numpy()

model.compile(optimizer='adam', loss=loss)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
#checkpoint_callback = ModelCheckpoint('./models/model', monitor='loss', save_best_only=True)

EPOCHS = 60

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

model.save_weights(filepath='weights')

model.summary()