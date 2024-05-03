import tensorflow as tf
import numpy as np


def positional_encoding(length, depth):
    ## REFERENCE: https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer
    ## TODO: Can remove signature
    depth = depth / 2
    ## Generate a range of positions and depths
    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
    ## Compute range of radians to take the sine and cosine of.
    angle_rates = 1 / (10000**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    ## This serves as offset for the Positional Encoding
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size

        ## TODO: Implement Component

        ## Embed labels into an optimizable embedding space
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)

        ## Implement sinosoidal positional encoding: offset by varying sinosoidal frequencies.
        ## HINT: May want to use the function above...
        self.pos_encoding = positional_encoding(length=window_size, depth=embed_size)

    def call(self, x):
        ## TODO: Get embeddings and and scale them by sqrt of embedding size, and add positional encoding.
        length = tf.shape(x)[1]
        embedded_input = self.embedding(x)
        scaled_input = embedded_input * tf.math.sqrt(
            tf.cast(self.embed_size, tf.float32)
        )
        pos_encoding = self.pos_encoding[:length, :]
        out = scaled_input + pos_encoding
        return out
