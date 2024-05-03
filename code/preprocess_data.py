import pickle
import random
import re
from PIL import Image
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import collections
from tqdm import tqdm
import pandas as pd
import librosa

def preprocess_captions(captions, window_size):
    for i, caption in enumerate(captions):
        # Taken from:
        # https://towardsdatascience.com/image-captions-with-attention-in-tensorflow-step-by-step-927dad3569fa

        # Convert the caption to lowercase, and then remove all special characters from it
        caption_nopunct = re.sub(r"[^a-zA-Z0-9]+", ' ', caption.lower())
      
        # Split the caption into separate words, and collect all words which are more than 
        # one character and which contain only alphabets (ie. discard words with mixed alpha-numerics)
        clean_words = [word for word in caption_nopunct.split() if ((len(word) > 1) and (word.isalpha()))]
      
        # Join those words into a string
        caption_new = ['<start>'] + clean_words[:window_size-1] + ['<end>']
      
        # Replace the old caption in the captions list with this new cleaned caption
        captions[i] = caption_new

def get_audio_features(audio_names, data_folder):
    '''
    Method used to extract the features from the images in the dataset using ResNet50
    '''
    audio_features = []
    #resnet = tf.keras.applications.ResNet50(False)  ## Produces Bx7x7x2048
    #gap = tf.keras.layers.GlobalAveragePooling2D()  ## Produces Bx2048
    pbar = tqdm(audio_names)
    for i, audio_name in enumerate(pbar):
        aud_path = f'{audio_name}'
        pbar.set_description(f"[({i+1}/{len(audio_names)})] Processing '{aud_path}'")
        #audio_data = tfio.audio.AudioIOTensor(aud_path)
        #audio_t = tf.squeeze(audio_data, axis=[-1])
        #normalized_audio = audio_t / tf.math.reduce_max(tf.math.abs(audio_t))
        #tensor = tf.cast(audio_t, tf.float32) / 32767.0
        #tensor = tensor.to_tensor()
        #img_in = tf.keras.applications.resnet50.preprocess_input(img_array)[np.newaxis, :]
        y, sr = librosa.load(aud_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        #print(f'MFCC: {mfccs.shape}')
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        #print(f'chroma: {chroma.shape}')
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        #print(f'tempo: {tempo.shape}, beats" {beats.shape}')
        combined_features = np.concatenate((mfccs, chroma), axis=0)

        mean = np.mean(combined_features, axis=0)
        std = np.std(combined_features, axis=0)
    
        normalized_features = (combined_features - mean) / (std + 1e-8) 
        padded = tf.keras.preprocessing.sequence.pad_sequences(normalized_features, maxlen = 1873, dtype='float32', padding='post', truncating='post')[np.newaxis, :]
        audio_features += [padded]
        #print(padded.shape)
    print()
    return audio_features


def load_data(data_folder):
    '''
    Method that was used to preprocess the data in the data.p file. You do not need 
    to use this method, nor is this used anywhere in the assignment. This is the method
    that the TAs used to pre-process the Flickr 8k dataset and create the data.p file 
    that is in your assignment folder. 

    Feel free to ignore this, but please read over this if you want a little more clairity 
    on how the images and captions were pre-processed 
    '''
    
    df = pd.read_csv('../data/musiccaps-files.csv')
    df = df[df['download_status'] == True][['caption','audio']]

    audio_to_caption_dict = {}
    for index, row in df.iterrows():
        audio_to_caption_dict[row['audio']] = row['audio']

    shuffled_audio = list(audio_to_caption_dict.keys())
    random.seed(0)
    random.shuffle(shuffled_audio)
    test_audio_names = shuffled_audio[:1000]
    train_audio_names = shuffled_audio[1000:]

    def get_all_captions(audio_names):
        to_return = []
        for audio in audio_names:
            captions = audio_to_caption_dict[audio]
            to_return.append(captions)
        return to_return


    # get lists of all the captions in the train and testing set
    train_captions = get_all_captions(train_audio_names)
    test_captions = get_all_captions(test_audio_names)

    #remove special charachters and other nessesary preprocessing
    window_size = 20
    preprocess_captions(train_captions, window_size)
    preprocess_captions(test_captions, window_size)

    # count word frequencies and replace rare words with '<unk>'
    word_count = collections.Counter()
    for caption in train_captions:
        word_count.update(caption)

    def unk_captions(captions, minimum_frequency):
        for caption in captions:
            for index, word in enumerate(caption):
                if word_count[word] <= minimum_frequency:
                    caption[index] = '<unk>'

    unk_captions(train_captions, 50)
    unk_captions(test_captions, 50)

    # pad captions so they all have equal length
    def pad_captions(captions, window_size):
        for caption in captions:
            caption += (window_size + 1 - len(caption)) * ['<pad>'] 
    
    pad_captions(train_captions, window_size)
    pad_captions(test_captions,  window_size)

    # assign unique ids to every work left in the vocabulary
    word2idx = {}
    vocab_size = 0
    for caption in train_captions:
        for index, word in enumerate(caption):
            if word in word2idx:
                caption[index] = word2idx[word]
            else:
                word2idx[word] = vocab_size
                caption[index] = vocab_size
                vocab_size += 1
    for caption in test_captions:
        for index, word in enumerate(caption):
            caption[index] = word2idx[word] 
    
    print("Getting training embeddings")
    train_audio_features = get_audio_features(train_audio_names, data_folder)
    print("Getting testing embeddings")
    test_audio_features = get_audio_features(test_audio_names, data_folder)
    #print(test_audio_features[0].shape)


    return dict(
        train_captions          = np.array(train_captions),
        test_captions           = np.array(test_captions),
        train_audio_features    = np.array(train_audio_features),
        test_audio_features     = np.array(test_audio_features),
        word2idx                = word2idx,
        idx2word                = {v:k for k,v in word2idx.items()},
    )


def create_pickle(data_folder):
    with open(f'{data_folder}/data.p', 'wb') as pickle_file:
        pickle.dump(load_data(data_folder), pickle_file)
    print(f'Data has been dumped into {data_folder}/data.p!')


if __name__ == '__main__':
    ## Download this and put the Images and captions.txt indo your ../data directory
    ## Flickr 8k Dataset: https://www.kaggle.com/datasets/adityajn105/flickr8k?resource=download
    data_folder = '../data'
    create_pickle(data_folder)