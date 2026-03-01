import tensorflow as tf
import numpy as np

# Load dataset
text = open("dataset.txt", "r", encoding="utf-8").read()

# Create character vocabulary
vocab = sorted(set(text))
char_to_idx = {u:i for i, u in enumerate(vocab)}
idx_to_char = np.array(vocab)

# Convert text to numbers
text_as_int = np.array([char_to_idx[c] for c in text])

# Parameters
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
dataset = dataset.shuffle(10000).batch(64, drop_remainder=True)

# Build RNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(vocab), 256),
    tf.keras.layers.LSTM(512, return_sequences=True),
    tf.keras.layers.Dense(len(vocab))
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

model.fit(dataset, epochs=10)

model.save("text_generator_model.h5")
