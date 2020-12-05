## -----------------------------------------------------------------------------
### Goal: Using speicfied weights file, reload Decoder and run inferences using the encodings.
##        Calculate the Loss.
##        NOTES:
##              1) The encoder is a Keras Google Inception-v3 model pre-trained on Imagenet. Encodings picked already.
##              2) Decoder is setup as coded below and the weights are populated from a locally saved file.
##              3) The Word2Index and Index2Word data structure are reloaded from pickle file locations.
##                 Also. the location of the image encodings file.
##                 These are hard-coded - CHANGE IF REQUIRED BEFORE RUNNING.
##
## -----------------------------------------------------------------------------
## Command line arguments:
## 1) -wtfile      : weights file for the decoder
## -----------------------------------------------------------------------------
## Usage example:
##    python3 ImgCap_Check_Val_Loss_Lappy_1.py -wtfile "/Path/PickleFile/WeightsDecoder/filename.h5"
## -----------------------------------------------------------------------------

from __future__ import print_function

import os
import json
import time
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
#from tensorflow.keras import layers
#from keras.preprocessing.sequence import pad_sequences
#import sys
#import copy
#import re
#import string
#import matplotlib.pyplot as plt
#import itertools
#import PIL
#import PIL.Image

import argparse

argparser = argparse.ArgumentParser(
    description='perform image captioning on given image')

argparser.add_argument(
    '-wtfile',
    '--decoder_weights_file_input',
    type=str,
    help='absolute path of the weights file to setup decoder')

def reload_rnn_encoder_saved_weights(_saved_weights_file, _EMBEDDING_DIMS, _VOCAB_SIZE, _MAX_LENGTH_CAPTION, _DEBUG_SWITCH = False):
    if os.path.exists(_saved_weights_file) and os.path.isfile(_saved_weights_file):
        ## Decoder Model defining
        
        ## parameters to define model
        #EMBEDDING_DIMS is initialised earlier while creating embedding matrix
        #VOCAB_SIZE is initialised earlier
        #MAX_LENGTH_CAPTION is initialised earlier
        
        inputs1 = keras.Input(shape=(2048,))
        fe1 = keras.layers.Dropout(0.5)(inputs1)
        fe2 = keras.layers.Dense(256, activation='relu')(fe1)
        
        # partial caption sequence model
        inputs2 = keras.Input(shape=(_MAX_LENGTH_CAPTION,))
        se1 = keras.layers.Embedding(_VOCAB_SIZE, _EMBEDDING_DIMS, mask_zero=True)(inputs2)
        se2 = keras.layers.Dropout(0.5)(se1)
        se3 = keras.layers.LSTM(256)(se2)
        
        # decoder (feed forward) model
        decoder1 = keras.layers.add([fe2, se3])
        decoder2 = keras.layers.Dense(256, activation='relu')(decoder1)
        outputs = keras.layers.Dense(_VOCAB_SIZE, activation='softmax')(decoder2)
        
        # merge the two input models
        reloaded_rnn_decoder_model = keras.models.Model(inputs=[inputs1, inputs2], outputs=outputs)
        
        ## load the weights
        reloaded_rnn_decoder_model.load_weights(_saved_weights_file)
        ## freeze all the layers
        reloaded_rnn_decoder_model.trainable = False

        print(f"\nSUCCESS - Reloaded weights from :: {_saved_weights_file}\n")
        print(f"RNN Decoder model (non-trainable type) defined with these paramenters:\nEMBEDDING_DIMS = {_EMBEDDING_DIMS} , VOCAB_SIZE = {_VOCAB_SIZE} , MAX_LENGTH_CAPTION = {_MAX_LENGTH_CAPTION}\nAttempting to load weights...")

        return reloaded_rnn_decoder_model
    else:
        print(f"\nERROR reloading weights. Check weights file exists here = {_saved_weights_file} ;\nOR model setup parameters incompatible with the saved weights file given.")
        return None

"""
## DATA GENERATOR FUNCTION USED DURING TRAINING - FOR REFERENCE
# data generator, use during the call to model.fit_generator() to create batchwise data
def data_generator_2(_descriptions, _imgs_features_arr, _wordtoix, _max_length, _images_per_batch, _vocab_size):
    X1, X2, y = [] , [] , []  ## empty lists to populate the input and target data for a bath
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in _descriptions.items():
            n+=1
            # retrieve the encoded features of image
            img_feat = _imgs_features_arr[ key ] # + '.jpg' ]  ## I am not using the .jpg in the key of image features array
            for desc in desc_list:
                # encode the sequence
                seq = [_wordtoix[word] for word in desc.split(' ') if word in _wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=_max_length)[0]
                    # encode output sequence
                    out_seq = keras.utils.to_categorical([out_seq], num_classes=_vocab_size)[0]
                    # store
                    X1.append(img_feat)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n == _images_per_batch:
                ## ValueError: No gradients provided for any variable: ['dense_2/kernel:0', 'dense_2/bias:0', 'lstm_1/lstm_cell_1/kernel:0', 'lstm_1/lstm_cell_1/recurrent_kernel:0', 'lstm_1/lstm_cell_1/bias:0', 'dense_3/kernel:0', 'dense_3/bias:0', 'dense_4/kernel:0', 'dense_4/bias:0'].
                #yield [[np.array(X1), np.array(X2)], np.array(y)]
                yield [np.array(X1), np.array(X2)], np.array(y)
                X1, X2, y = list(), list(), list()
                n=0
"""
## using the same approach as data generator used during the decoder training - but now sending all data together
def create_data_for_evaluation(_descriptions, _imgs_features_arr, _wordtoix, _max_length, _vocab_size):
    X1, X2, y = [] , [] , []  ## empty lists to populate the input and target data for a bath

    for key, desc_list in _descriptions.items():
        # retrieve the encoded features of image
        img_feat = _imgs_features_arr[ key ] # + '.jpg' ]  ## I am not using the .jpg in the key of image features array
        for desc in desc_list:
            # encode the sequence
            seq = [_wordtoix[word] for word in desc.split(' ') if word in _wordtoix]
            # split one sequence into multiple X, y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=_max_length)[0]
                # encode output sequence
                out_seq = keras.utils.to_categorical([out_seq], num_classes=_vocab_size)[0]
                # store
                X1.append(img_feat)
                X2.append(in_seq)
                y.append(out_seq)
    return [np.array(X1), np.array(X2)], np.array(y)

def _main_(args):

    DEBUG_SWITCH = False
    
    # process command line arguments
    SAVED_WEIGHTS_PATH  = args.decoder_weights_file_input     # -wtfile      parameter

    BATCH_SIZE = 64

    IMG_ENCODINGS_PATH = r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Code/ModelsRuns/ImgCap/SavedData/Thesis_ImgCap_Run2_Deterministic_Info/img_encodings_val_3000.pkl'
    DESCRIPTIONS_PATH = r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Code/ModelsRuns/ImgCap/SavedData/Thesis_ImgCap_Run2_Deterministic_Info/descriptions_val_3000.pkl'
    I2W_FILE = r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Code/ModelsRuns/ImgCap/SavedData/Thesis_ImgCap_Run2_Deterministic_Info/ixtoword_train_97000.pkl'
    W2I_FILE = r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Code/ModelsRuns/ImgCap/SavedData/Thesis_ImgCap_Run2_Deterministic_Info/wordtoix_train_97000.pkl'

    ## basic sanity checks
    if (not os.path.exists(I2W_FILE) ) or (not os.path.isfile(I2W_FILE )):
        print(f"\nFATAL ERROR: Index to Word dict not found at :: {I2W_FILE}\nPremature Exit with exit code 110")
        exit(110)
    if (not os.path.exists(W2I_FILE) ) or (not os.path.isfile(W2I_FILE )):
        print(f"\nFATAL ERROR: Word to Index dict not found at :: {W2I_FILE}\nPremature Exit with exit code 111")
        exit(111)
    if (not os.path.exists(SAVED_WEIGHTS_PATH) ) or (not os.path.isfile(SAVED_WEIGHTS_PATH )):
        print(f"\nFATAL ERROR: Weights file for Decoder model not found :: {W2I_FILE}\nPremature Exit with exit code 112")
        exit(112)
    if (not os.path.exists(IMG_ENCODINGS_PATH) ) or (not os.path.isfile(IMG_ENCODINGS_PATH )):
        print(f"\nFATAL ERROR: Encodings file not found here :: {IMG_ENCODINGS_PATH}\nPremature Exit with exit code 113")
        exit(113)
    if (not os.path.exists(DESCRIPTIONS_PATH) ) or (not os.path.isfile(DESCRIPTIONS_PATH )):
        print(f"\nFATAL ERROR: Descriptions file not found here :: {DESCRIPTIONS_PATH}\nPremature Exit with exit code 114")
        exit(114)
    
    try:
        ## load the ixtoword and wordtoix dicts
        with open(I2W_FILE, 'rb') as handle:
            ixtoword = pickle.load(handle)
        with open(W2I_FILE, 'rb') as handle:
            wordtoix = pickle.load(handle)
        print(f"\nCheck wordtoix entries ::\nstartseq = {wordtoix.get('startseq')}\tendseq = {wordtoix.get('endseq')}\tbird = {wordtoix.get('bird')}")
        print(f"Check ixtoword entries ::\nix 1 = {ixtoword.get(1)}\tix 10 = {ixtoword.get(10)}\tix 974 = {ixtoword.get(974)}\n")
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Could not load the Word2Ix and/ or Ix2Word dicts from specified locations ::")
        print(f"Word2Ix :: {W2I_FILE}\nIx2Word :: {I2W_FILE}")
        print(f"Premature Exit with exit code 120")
        exit(120)
    except Exception as error_word_index_dict_load_msg:
        print(f"\nFATAL ERROR: unknown error loading the Word2Ix and/ or Ix2Word dicts :: {error_word_index_dict_load_msg}")
        print(f"Premature Exit with exit code 121")
        exit(121)

    try:
        ## the encodings file
        with open(IMG_ENCODINGS_PATH, 'rb') as handle:
            imgs_encodings_arr = pickle.load(handle)
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Could not load the saved Image Encodings from here ::\n{IMG_ENCODINGS_PATH}")
        print(f"Premature Exit with exit code 122")
        exit(122)
    except Exception as error_img_encodings_load_msg:
        print(f"\nFATAL ERROR: unknown error loading the saved Image Encodings :: {error_img_encodings_load_msg}")
        print(f"Premature Exit with exit code 123")
        exit(123)
    
    try:
        ## the descriptions file
        with open(DESCRIPTIONS_PATH, 'rb') as handle:
            descriptions_arr = pickle.load(handle)
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Could not load the Descriptions from here ::\n{DESCRIPTIONS_PATH}")
        print(f"Premature Exit with exit code 124")
        exit(124)
    except Exception as error_descriptions_load_msg:
        print(f"\nFATAL ERROR: unknown error loading the Descriptions :: {error_descriptions_load_msg}")
        print(f"Premature Exit with exit code 125")
        exit(125)
    
    ## lengths of the the two dicts MUST be equal
    if len(wordtoix) != len(ixtoword):
        print(f"\nFATAL ERROR: Lengths of Word2Ix and Ix2Word dicts were not equal.")
        print(f"Word2Ix length = {len(wordtoix)}\t\t Ix2Word length = {len(ixtoword)}")
        print(f"Premature Exit with exit code 130")
        exit(130)
    
    try:
        ### Load the Decoder network using the saved weights file
        ## Parameters to use while defining the Decoder model again
        EMBEDDING_DIMS = 200
        VOCAB_SIZE = 6758
        MAX_LENGTH_CAPTION = 49
        reloaded_RNN_decoder = reload_rnn_encoder_saved_weights(SAVED_WEIGHTS_PATH, EMBEDDING_DIMS, VOCAB_SIZE, MAX_LENGTH_CAPTION, DEBUG_SWITCH)
        if DEBUG_SWITCH:
            type(reloaded_RNN_decoder)   ## should say     tensorflow.python.keras.engine.training.Model
    except Exception as error_decoder_load_msg:
        print(f"\nFATAL ERROR: Could not load LSTM-Decoder model.")
        print(f"Error message :: {error_decoder_load_msg}\nPremature Exit with exit code 140")
        exit(140)
    
    ## DEBUG TESTING ONLY
    if DEBUG_SWITCH:
        ## keep just a few items for testing out
        descriptions_arr = dict(list(descriptions_arr.items())[:4])
        print(f"\nReduced Descriptions:\nlength = {len(descriptions_arr)}\ndescriptions_arr_less =\n{descriptions_arr}\n")
    
    print(f"\nLength of Descriptions dict = {len(descriptions_arr)}\n")

    ## compile model - note that all layers frozen as model set as trainable = False during loading
    reloaded_RNN_decoder.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    
    ## make data suitable to use for model evaluation step
    inputs, outputs = create_data_for_evaluation(descriptions_arr, imgs_encodings_arr, wordtoix, MAX_LENGTH_CAPTION, VOCAB_SIZE)

    start_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_tick = time.time()

    print(f"\n\nStarted at = {start_timestamp}\n")

    ## evaluate it and get score
    model_loss = reloaded_RNN_decoder.evaluate(inputs, outputs, batch_size=BATCH_SIZE)
    print(f"\n\nLoss with Batch size of {BATCH_SIZE} =\n{model_loss}\n\n")

    end_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    end_tick = time.time()

    print(f"\n\nEnded at = {end_timestamp}\nTime taken = {end_tick - start_tick} seconds\n")
    
    return

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
    print(f"\nDone\n")