import tensorflow as tf
import keras_nlp

from positional_encoding import PositionalEncoding


class TransformerDecoder(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # Define feed forward layer(s) to embed audio features into a vector
        self.audio_embedding = tf.keras.layers.Dense(
            self.hidden_size, activation="relu"
        )

        # Define positional encoding to embed and offset layer for language:
        self.encoding = PositionalEncoding(vocab_size, hidden_size, window_size)

        # Define transformer decoder layer:
        self.decoder = keras_nlp.layers.TransformerDecoder(hidden_size, 5)

        # Define classification layer(s) (LOGIT OUTPUT)
        self.classifier = tf.keras.layers.Dense(self.vocab_size)

    def call(self, encoded_audio, captions):
        audio_enc = self.audio_embedding(tf.squeeze(encoded_audio, axis=1))
        captions_enc = self.encoding(captions)
        decoder_out = self.decoder(captions_enc, audio_enc)
        logits = self.classifier(decoder_out)
        return logits
