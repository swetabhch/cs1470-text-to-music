import os
import argparse
import numpy as np
import pickle
import tensorflow as tf
from typing import Optional
from types import SimpleNamespace


from model import AudioCaptionModel, accuracy_function, loss_function
from decoder import TransformerDecoder

# from positional_encoding import PositionalEncoding


def parse_args(args=None):
    """
    Perform command-line argument parsing (other otherwise parse arguments with defaults).
    To parse in an interative context (i.e. in notebook), add required arguments.
    These will go into args and will generate a list that can be passed in.
    For example:
        parse_args('--type', 'rnn', ...)
    """
    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=["rnn", "transformer"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--task", required=True, choices=["train", "test", "both"], help="Task to run"
    )
    parser.add_argument(
        "--data", required=True, help="File path to the assignment data file."
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs used in training."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Model's learning rate")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "rmsprop", "sgd"],
        help="Model's optimizer",
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="Model's batch size."
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="Hidden size used to instantiate the model.",
    )
    parser.add_argument(
        "--window_size", type=int, default=20, help="Window size of text entries."
    )
    parser.add_argument(
        "--chkpt_path", default="", help="where the model checkpoint is"
    )
    parser.add_argument(
        "--check_valid",
        default=True,
        action="store_true",
        help="if training, also print validation after each epoch",
    )
    if args is None:
        return parser.parse_args()  ## For calling through command line
    return parser.parse_args(args)  ## For calling through notebook.


def main(args):

    ##############################################################################
    ## Data Loading
    with open(args.data, "rb") as data_file:
        data_dict = pickle.load(data_file)

    train_captions = data_dict["train_captions"]
    test_captions = data_dict["test_captions"]
    # TODO: figure out what kind of feat prep we need to do if any
    #    if we're talking about number of captions, we need to do none.
    train_audio_feats = data_dict["train_audio_features"]
    test_audio_feats = data_dict["test_audio_features"]
    word2idx = data_dict["word2idx"]

    # feat_prep = lambda x: np.repeat(np.array(x).reshape(-1, 2048), 5, axis=0)
    # # img_prep  = lambda x: np.repeat(x, 5, axis=0)
    # # train_images    = img_prep(data_dict['train_images'])
    # # test_images     = img_prep(data_dict['test_images'])
    # # idx2word        = data_dict['idx2word']

    ## Training task
    if args.task in ("train", "both"):
        decoder = TransformerDecoder(len(word2idx), args.hidden_size, args.window_size)
        model = AudioCaptionModel(decoder)
        compile_model(model, args)
        train_model(
            model,
            train_captions,
            train_audio_feats,
            word2idx["<pad>"],
            args,
            valid=(test_captions, test_audio_feats),
        )
        if args.chkpt_path:
            ## Save model to run testing task afterwards
            save_model(model, args)

    ## Testing task
    if args.task in ("test", "both"):
        if args.task != "both":
            model = load_model(args)
        if not (args.task == "both" and args.check_valid):
            test_model(model, test_captions, test_audio_feats, word2idx["<pad>"], args)


##############################################################################
## UTILITY METHODS


def save_model(model, args):
    """Loads model based on arguments"""
    os.makedirs(f"{args.chkpt_path}", exist_ok=True)

    tf.keras.models.save_model(model, args.chkpt_path)
    print(f"Model saved to {args.chkpt_path}")


def load_model(args):
    """Loads model by reference based on arguments. Also returns said model"""
    model = tf.keras.models.load_model(
        args.chkpt_path,
        custom_objects=dict(
            TransformerDecoder=TransformerDecoder,
            AudioCaptionModel=AudioCaptionModel,
        ),
    )

    ## Saving is very nuanced. Might need to set the custom components correctly.
    ## Functools.partial is a function wrapper that auto-fills a selection of arguments.
    ## so in other words, the first argument of AudioCaptionModel.test is model (for self)
    from functools import partial

    model.test = partial(AudioCaptionModel.test, model)
    model.train = partial(AudioCaptionModel.train, model)
    model.compile = partial(AudioCaptionModel.compile, model)
    compile_model(model, args)
    print(f"Model loaded from '{args.chkpt_path}'")
    return model


def compile_model(model, args):
    """Compiles model by reference based on arguments"""
    optimizer = tf.keras.optimizers.get(args.optimizer).__class__(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy_function])


def train_model(model, captions, audio_feats, pad_idx, args, valid):
    """Trains model and returns model statistics"""
    stats = []
    try:
        for epoch in range(args.epochs):
            stats += [
                model.train(captions, audio_feats, pad_idx, batch_size=args.batch_size)
            ]
            if args.check_valid:
                model.test(valid[0], valid[1], pad_idx, batch_size=args.batch_size)
    except KeyboardInterrupt as e:
        if epoch > 0:
            print(
                "Key-value interruption. Trying to early-terminate. Interrupt again to not do that!"
            )
        else:
            raise e

    return stats


def test_model(model, captions, audio_feats, pad_idx, args):
    """Tests model and returns model statistics"""
    perplexity, accuracy = model.test(
        captions, audio_feats, pad_idx, batch_size=args.batch_size
    )
    return perplexity, accuracy


## END UTILITY METHODS
##############################################################################

if __name__ == "__main__":
    main(parse_args())
