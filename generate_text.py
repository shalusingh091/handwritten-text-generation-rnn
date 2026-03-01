import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model("text_generator_model.h5")

text = open("dataset.txt", "r", encoding="utf-8").read()
vocab = sorted(set(text))
char_to_idx = {u:i for i, u in enumerate(vocab)}
idx_to_char = np.array(vocab)

def generate_text(model, start_string, num_generate=300):

    input_eval = [char_to_idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    temperature = 1.0

    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = predictions[:, -1, :]
        predictions = predictions / temperature

        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx_to_char[predicted_id])

    return start_string + ''.join(text_generated)

print(generate_text(model, start_string="Hello "))
