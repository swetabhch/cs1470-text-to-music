import tensorflow as tf
import keras_nlp

from positional_encoding import PositionalEncoding


class TransformerDecoder(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # Define feed forward layer(s) to embed image features into a vector
        self.audio_embedding = tf.keras.layers.Dense(
            self.hidden_size, activation="relu"
        )

        # Define positional encoding to embed and offset layer for language:
        self.encoding = PositionalEncoding(vocab_size, hidden_size, window_size)

        # Define transformer decoder layer:
        # self.decoder = TransformerBlock(hidden_size)
        self.decoder = keras_nlp.layers.TransformerDecoder(hidden_size,1)

        # Define classification layer(s) (LOGIT OUTPUT)
        self.classifier = tf.keras.layers.Dense(self.vocab_size)

    def call(self, encoded_audio, captions):
        # TODO:
        # 1) Embed the encoded images into a vector (HINT IN NOTEBOOK)
        # 2) Pass the captions through your positional encoding layer
        # 3) Pass the english embeddings and the image sequences to the decoder
        # 4) Apply dense layer(s) to the decoder out to generate **logits**
        audio_enc = self.audio_embedding(tf.squeeze(encoded_audio, axis=1))
        captions_enc = self.encoding(captions)
        decoder_out = self.decoder(captions_enc, audio_enc)
        logits = self.classifier(decoder_out)
        return logits
