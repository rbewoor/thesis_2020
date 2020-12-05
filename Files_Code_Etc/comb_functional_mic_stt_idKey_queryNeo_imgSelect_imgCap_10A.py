## -----------------------------------------------------------------------------
###             FUNCTIONAL programming approach
### Goal: Accept user input via mic and save each of the three inputs as different Wav files.
###          NOTE: Mic input is optional and user can choose to run program by executing from Stage 2 directly.
###       Process wav files thru Speech to Text to get transcriptions for each of the three input files.
###       Do NLP processing on the transcriptions to extract the keywords (objects of interest).
###       Then query the Neo4j database to return images containing the objects of interest.
###          Per query, limit results to maximum 20 images.
###          These are the "candidate images".
###       Present candidate images via GUI to user to inspect and/or deselect images.
###          Per Query, maximum of 5 images can be Selected.
###          Thus reduce the total images from up to 60 (20 x 3), to maximum 15 images.
###          NOTE: For some query, user may Deselect ALL the images if none make sense.
###       Then perform Image Captioning on the selected images.
###          Two models used here:
###              Model 1: Merge  type No-attention   RNN (LSTM) model with Beam search (width=3).
###              Model 2: Inject type With-attention RNN (GRU)  model with Greedy search.
###       Present the images and captions via GUI to user to inspect, edit captions and deselect
###          any images.
###          NOTE: Again user may deselect all images for a query if none make sense.
###       Finally, save a data structure of the results till this point to file
###          for processing later by the Story Geneation logic.
## -----------------------------------------------------------------------------
##  STAGE 1) Mic Input and STT Inference - with GUI:
##     Accept 3 sentences as user input from mic (optional).
##        Else, no mic input, but process the 3 already saved wav files.
##     Get the transcriptions by running Speech-to-Text processing on the Wav files.
##        Show the transcriptions without option to edit.
##     Do the processing via a GUI.
##     Replace certain words for downstream processing. Examples:
##         Output of Deespeech                                      Output after replacing words
##         a person carries hand bag in left hand                   a person carries handbag in left hand
##         the television monitor is too close to you               the tvmonitor is too close to you
##     Replacement required as the database is populated with specific object labels. If words are not corrected,
##         then Id Key Elements will not successfully match against those labels and the words will be ignored.
##     Outputs:
##        a) Intermediate data for the next stage for downstream processing.
##        b) If running with Mic input:
##              b1) 3 wav files to be stored in CLA parameter location
##              b2) Create the file specified by CLA parameter to hold the absolute paths of the 3 above Wav files.
##        c) If NOT running with Mic input:
##              Nothing new created as the wav files exist already.
## 
##  STAGE 2) Identify keywords and Deselect words via GUI
##        Process the input data which is the speech to text transcriptions.
##        Using Spacy, remove stop-words, perform part of speech tagging.
##        Write all POS info for these non stopwords to a file (word types as noun or verb).
##        But for the candidate keywords pick up only the nouns.
##        Using a GUI present the candidate keywords to user for Deselecting words.
##             Ensure only valid number of keywords are Selected per input sentence.
##        Pass the final data structure of Selected words for processing by database query module.
##     Outputs:
##        a) One text file:
##           For ALL words that are not Stopwords, of type Noun or Verb, their 
##               comprehensive nlp info from Spacy.
##        b) Intermediate data:
##           The data strucutre required for neo4j query logic.
##           Depending on the value of the boolean variable POS_INTEREST_INCLUDE_VERBS_SWITCH
##              either only nouns, or both nouns and verbs, will be retained. Then the user
##              drops any words if required.
## 
##  STAGE 3) Query Neo4j for images matching the keywords
##     Query Neo4j database to find images that contain the objects of interest.
## 
##     Neo4j database schema:
##             (:Image{name, dataset}) - HAS{score} -> (:Object{name})
##             Image Nodes:
##             - name property is the name of the image file  
##             - dataset property is to track the source of the image:
##               e.g. "coco_val_2017" for images from COCO validation 2017 dataset
##               e.g. "coco_test_2017" for images from COCO test 2017 dataset
##               e.g. "flickr30" for images from Flickr 30,000 images dataset
##             Object Nodes:
##             - name property is the label of the object from the object detector.
##             - datasource property used to keep track of which dataset this image belongs to
##             HAS relationship:
##             - score property is confidence score of the object detector. Value can theoretically be from 0.00 to 100.00
## 
##      Currently, only these objects are in the database - SORTED ALPHABETICALLY FOR READABILITY
##         NOTE: The actual order used in code is important as the pretrained model expects labels in that order only
##         labels = ['aeroplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', \
##                   'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', \
##                   'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', \
##                   'clock', 'cow', 'cup', 'diningtable', 'dog', 'donut', 'elephant', 'fire hydrant', \
##                   'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', \
##                   'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorbike', 'mouse', \
##                   'orange', 'oven', 'parking meter', 'person', 'pizza', 'pottedplant', \
##                   'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis', \
##                   'snowboard', 'sofa', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', \
##                   'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', \
##                   'train', 'truck', 'tvmonitor', 'umbrella', 'vase', 'wine glass', 'zebra']
## 
##  STAGE 4) GUI based Deselection of candidate images obtained from Neo4j query stage.
##        Allow user to Enlarge an image, perform object detection inference and finally DESELECT images.
##              This is because some images may have incorrectly found the objects of interest.
##              Each Neo4j query can return up to 20 images. But the maximum number of images per query to
##                   pass to the next stage (Auto caption block) is limited to 5 per query. So allow culling.
##        Logic:
##           a) Show the up to 20 images in a grid pattern as thumbnails.
##           b) User can click on image thumbnail to Select or Deslect an image.
##              By default, initially all the images are Selected.
##           c) Display the current count of Selected images.
##              Change the color of this box (Label) to Green or Red to indicate if within max. limit.
##           d) Provide option to Enlarge image to study more clearly.
##           e) Provide option to perform object detection inference again on an image and see results.
##           f) Allow Confirmation of Deselections by cliking the Confirm butoon.
##              Allow clicking only if current count of Selections <= max limit.
##       Outputs:
##          None
##              An intermediate file is written and read during logic.
##              Will be automatically deleted as part of execution flow.
## 
##  STAGE 5A) Image captioning with Merge No-Attention model on the images selected by the user.
##        Setup a the Encoder-Decoder model to perform inference. Insert predicted captions in data structure.
##        Logic:
##           a) The encoder is a Keras Google Inception-v3 model pre-trained on Imagenet.
##           b) Decoder is a 256 cell LSTM with merge architecture.
##                  Weights are populated from a locally saved file.
##           c) The Decoder is trained with the following characteristics and hence model is defined using these parameters:
##                   i)   EMBEDDING_DIMS = 200
##                   ii)  VOCAB_SIZE = 6758 (occurrence frequency in full vocabulary > 10 times)
##                   iii) MAX_LENGTH_CAPTION = 49
##           d) The Word2Index and Index2Word data structure are reloaded from pickle file locations.
##              These are hard-coded - CHANGE IF REQUIRED BEFORE RUNNING.
##           e) If in debug mode, each image will be briefly displayed before proceeding to generate inference
##              using the decoder.
##           f) Expects the results from two previous stages to be sent. This will be combined for processing by
##              this stage. The two previous stages are:
##                   i)  Id Key Elements - final selected words by user for each sentence that were used to 
##                       query Neo4j.
##                   ii) GUI Candidate Image Selection - final images selected by user from the images output by 
##                       Neo4j query.
##           g) Uses Beam Search for predictions. Variable at start of Control logic specifies value (currently 5)
##       Outputs:
##           a) None
## 
##  STAGE 5B) Image captioning with Inject With-Attention model on the images selected by the user.
##        Setup a the Encoder-Decoder model to perform inference. Insert predicted captions in data structure.
##        Logic:
##           a) The encoder is a Keras Google Inception-v3 model pre-trained on Imagenet (same as before).
##           b) Decoder is a 512 cell GRU with inject architecture and 64 sized Badhanau attention.
##                  Model restored from checkpoint files - not from a weights file.
##           c) The Decoder is trained with the following characteristics and hence model is defined using these parameters:
##                   i)   EMBEDDING_DIMS = 256
##                   ii)  VOCAB_SIZE = 5000 (most frequent words in full vocabulary)
##                   iii) MAX_LENGTH_CAPTION = 52
##                   iv)  Attention size = 64
##           d) Updates the data structure created by the No-Attention image caption logic.
##           g) Uses Greedy Search for predictions.
##       Outputs:
##           a) None
## 
##  STAGE 6) GUI presentation of the Images and their captions.
##        Show the images and both version of their captions to the user.
##        Allows option to edit the captions for corrections.
##        Logic:
##           a) One main window for each original input sentence with instructions.
##           b) Subwindow shows the images and their captions.
##              Button here to Enlarge Image + Edit Caption.
##              Independant of any editing to caption, the images can be deselected by user to not pass on the
##                  next stage i.e. Story Generation logic
##           c) Subwindow to Enlarge Image and/or Edit Caption
##       Outputs:
##           a) One text file:
##              Contains the final data structure of the key words, the images and the two versions of optionally edited captions.
##              This file picked up as input by the Story Generator logic.
##
## -----------------------------------------------------------------------------
## Command line arguments:
## 1) -micYorN          : Wwitch to bypass/include mic input stage execution.
## 2) -sw_here          : If mic input stage executed, then the newly created wav files ot be saved in this directory.
##                        But if mic input stage is bypassed, then irrelevant.
## 3) -file4STT         : This is the file used by STT stage for inference. Expects each line to have the path for one file. Therefore, three lines in file expected.
##                        If mic input stage is executed, then this file will be created (with entries of absolute paths of the newly created wav files).
##                        But if mic input stage is bypassed, then this file MUST already exist with required three entries with the three wav files existing at those locations specified.
## 4) -opfileallposinfo : Location where to write a file with parts-of-speech (POS) info during Identify key elements stage.
## 5) -file4stygen      : Location to save the results after GUI display (and possible editing) of Image captions. Will be used by
##                        story generator logic as input file.
## 6) -logfileloc       : Location to write the log file.
## -----------------------------------------------------------------------------
## Usage example:
## 
## OPTION 1 - Running with mic input. Specify micYorN parameter = Y.
## python3 comb_functional_stt_id_gui_query_img_select_gui_img_cap_6D-WIP.py -micYorN "Y" -sw_here "/home/rohit/PyWDUbuntu/thesis/audio/wavs/fromMic/" -file4STT "/home/rohit/PyWDUbuntu/thesis/combined_execution/SttTranscribe/mic_input_wavfiles_1.txt" -opfileallposinfo "/home/rohit/PyWDUbuntu/thesis/combined_execution/IdElements/all_words_pos_info_1.txt" -logfileloc "./LOG_comb_functional_stt_id_gui_query_img_select_gui_img_cap_6D_WIP.LOG"
## 
## OPTION 2 - Bypass mic input logic, directly execute STT inference logic. Specify micYorN parameter = N.
##            Note that file4STT parameter file MUST exist already and the wav files should be present in the locations specified in this file.
## python3 comb_functional_stt_id_gui_query_img_select_gui_img_cap_6D-WIP.py -micYorN "N" -sw_here "/home/rohit/PyWDUbuntu/thesis/audio/wavs/fromMic/" -file4STT "/home/rohit/PyWDUbuntu/thesis/combined_execution/SttTranscribe/mic_input_wavfiles_1.txt" -opfileallposinfo "/home/rohit/PyWDUbuntu/thesis/combined_execution/IdElements/all_words_pos_info_1.txt" -logfileloc "./LOG_comb_functional_stt_id_gui_query_img_select_gui_img_cap_6D_WIP.LOG"
## -----------------------------------------------------------------------------

## import necessary packages

##   imports for common or generic use packages
from __future__ import print_function
import argparse
import os
import json
import copy
import time
import datetime
import string
import pickle
import numpy as np
import tkinter as tk
from functools import partial
import shutil
import PIL
import logging

##   imports for mic input and save as wav files
import pyaudio
import wave
import contextlib

##   imports for stt logic
import subprocess

##   imports for identify keywords logic
import spacy
#from spacy.lang.en.stop_words import STOP_WORDS - not required as using alternative approach

##   imports for neo4j query logic
import sys
from py2neo import Graph

##   imports for gui selection of images returned by neo4j query stage
from PIL import ImageTk, Image
from keras.models import Model as keras_Model, load_model as keras_load_model
import struct
import cv2

##   imports for image captioning inference - no gui stage
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

##   imports for gui display of image captioning inference
## nothing new to import

## command line arguments
argparser = argparse.ArgumentParser(
    description='parameters to run this program')

argparser.add_argument(
    '-micYorN',
    '--mic_or_saved_wavs',
    help='flag to indicate whether user input via microphone or via saved wav files')

argparser.add_argument(
    '-sw_here',
    '--save_wavs_here',
    help='folder where to save the recordings from microphone')

argparser.add_argument(
    '-file4STT',
    '--file_wav_loc_for_stt',
    help='file with each line containing absolute paths of each wav file for STT to process')

argparser.add_argument(
    '-opfileallposinfo',
    '--opfileallwordsposinfo',
    help='location for output file where the key elements will be stored')

argparser.add_argument(
    '-file4stygen',
    '--opfile_for_sty_gen',
    help='location to save results for story generator logic to process')

argparser.add_argument(
    '-logfileloc',
    '--oplogfilelocation',
    help='location for output file for logging')


###############################################################################################################
## -----------------------  MIC + STT LOGIC STARTS              MIC + STT LOGIC  STARTS ----------------------- 
###############################################################################################################

class c_micSttRootWindow():
    def __init__(self, _DEBUG_SWITCH, _micYorN, _save_wav_here, _wavlocfile, _ds_infs_arr, _wav_files_arr):        
        self.msg_warning_for_root_do_not_close = r'DO NOT CLOSE THIS WINDOW TILL YOU ARE READY TO MOVE TO NEXT STAGE!!!'
        self.msg_instructions_for_root = "".join([
            f"                               ----------------------------------",
            f"\n                               -----      INSTRUCTIONS      -----",
            f"\n                               ----------------------------------",
            f"\n",
            f"\n",
            f"\nThis is the main control window for Microphone input and STT inference.",
            f"\n",
            f"\n\t1.  Click on the button below to proceed.",
            f"\n\t       Important: Depending on the way you ran the program, you may be allowed to record."
            f"\n",
            f"\n\t2.  If microphone input IS REQUIRED, in the next window:",
            f"\n\t       * You MUST record three sentences.",
            f"\n\t       * Click the Record button and speak.",
            f"\n\t       * Click the Play   button to listen to your recording.",
            f"\n\t       * Click the Inference button for each file to do the Speech to Text",
            f"\n\t               processing and see the output transcription.",
            f"\n\t       * If a particular recording or its transcription is unsatisfactory, then you may",
            f"\n\t               record that file again and runs its STT processing again.",
            f"\n",
            f"\n\t3.  If NO microphone input is required, in the next window:",
            f"\n\t       * You MUST have provided an input file containing the three pre-recorded Wav files.",
            f"\n\t       * Click the Play button to listen to what has been recorded earlier.",
            f"\n\t       * You CANNOT record the file again.",
            f"\n\t       * Click the Inference button for each file to do the Speech to Text",
            f"\n\t               processing and see the output transcription.",
            f"\n",
            f"\n\t4.  Only after ALL three transcriptions are ready, click the Confirm button.",
            f"\n",
            f"\n\t5.  Once you return to this window, simply close the window to move to next stage.",
            f"\n"
        ])
        ## root window for the query being processed
        self.root = tk.Tk()
        self.DEBUG_SWITCH = _DEBUG_SWITCH
        self.micYorN = _micYorN
        self.save_wav_here = _save_wav_here
        self.wavlocfile = _wavlocfile
        self.wav_files_arr = _wav_files_arr
        self.ds_infs_arr = _ds_infs_arr
        self.root.title(f"Microphone input and STT processing - Click button to Proceed")
        ## label for warning message not to close
        self.lbl_root_msg_warning_not_close = tk.Label(
            master=self.root,
            text=self.msg_warning_for_root_do_not_close,
            bg="red", fg="white",
            width=(len(self.msg_warning_for_root_do_not_close) + 10),
            height=5
            )
        ## label for instructions
        self.lbl_root_instructions = tk.Label(
            master=self.root,
            text=self.msg_instructions_for_root,
            bg="blue", fg="white",
            justify=tk.LEFT,
            width=130,
            height=35,
            relief=tk.SUNKEN
            )
        ## button to proceed to subwindow
        self.btn_root_click_proceed = tk.Button(
            master=self.root,
            text=f"Click to Proceed",
            bg="yellow", fg="black",
            relief=tk.RAISED,
            borderwidth=15,
            command=partial(
                micStt_show_subwindow,
                self.DEBUG_SWITCH,
                self.root,
                self.micYorN,
                self.save_wav_here,
                self.wavlocfile,
                self.ds_infs_arr,
                self.wav_files_arr
                )
            )
        self.btn_root_click_proceed.configure(
            width=( len(self.btn_root_click_proceed["text"]) + 10 ),
            height=3
            )
        ## pack it all
        self.lbl_root_msg_warning_not_close.pack(padx=15, pady=15)
        self.lbl_root_instructions.pack(padx=10, pady=10)
        self.btn_root_click_proceed.pack(padx=50, pady=50)

class c_micStt_wnd_grid_window:
    def __init__(self, _DEBUG_SWITCH, _root, _micYorN, _save_wav_here, _wavlocfile, _ds_infs_arr, _wav_files_arr):
        self.root = _root
        self.micYorN = _micYorN
        self.save_wav_here = _save_wav_here
        self.wavlocfile = _wavlocfile
        self.ds_infs_arr = _ds_infs_arr
        self.wav_files_arr = _wav_files_arr
        self.DEBUG_SWITCH = _DEBUG_SWITCH
        ## audio related parameters- set as per requirement for Deepspeech = 16-bit, Mono channel, 16kHz
        self.DATA_APPEND_SECS = 3
        self.LISTEN_SECS = 10                 ## listen for only this much time, then stops recording
        self.SAMPLE_FORMAT = pyaudio.paInt16  ## 16 bit
        self.CHANNELS = 1                     ## mono channel
        self.FS = 16000                       ## sampling freq 16kHz
        self.CHUNCK = 1024   
        ## create skeleton command
        self.ds_inf_cmd_fixed = " ".join([
            f"deepspeech",
            f"--model /home/rohit/deepspeech/pretrained/v073/deepspeech-0.7.3-models.pbmm",
            f"--scorer /home/rohit/deepspeech/pretrained/v073/deepspeech-0.7.3-models.scorer",
            f"--audio",
            f""     ##/path/to/the/wavFile2Infer.wav - this last part will be added during runtime later
            ])
        #self.ds_inf_cmd_fixed = "deepspeech " + \
        #    "--model /home/rohit/deepspeech/pretrained/v073/deepspeech-0.7.3-models.pbmm " + \
        #        "--scorer /home/rohit/deepspeech/pretrained/v073/deepspeech-0.7.3-models.scorer " + \
        #            "--audio " ##/path/to/the/wavFile2Infer.wav - this last part will be added on the fly
        ## NOTE: The rows have these widgets:
        ##       1 - Play and Record Play buttons
        ##       2 to 4 - Labels for Paths of three files
        ##       5 - Buttons for Perform inference
        ##       6 to 8 - Labels for Inference rows
        ##
        self.n_rows = 8
        self.n_cols = 6
        ## 2-D array for the widgets of first 8 rows
        self.grid_buttons_arr = []
        ## window for the grid selection
        self.wnd_grid = tk.Toplevel(master=self.root)
        self.wnd_grid.title(f"Subwindow - Record audio and perform Speech to Text inference")
        ## widgets not part of grid
        ## label for messages to user
        self.lbl_user_msg_text = tk.StringVar()
        self.lbl_user_msg = tk.Label(
            master=self.wnd_grid,
            textvariable=self.lbl_user_msg_text,
            bg="blue", fg="white",
            borderwidth=10,
            relief=tk.SUNKEN
            )
        self.lbl_user_msg_text.set(f"Ready to Play or Record audio/ Perform inference" if self.micYorN == 'y' else f"Ready to Play audio/ Perform inference - No Recording allowed")
        self.lbl_user_msg.configure(
            width=( len(self.lbl_user_msg["text"]) + 10 ),
            height=5
            )
        ## button for selection confirm - always enabled
        self.btn_confirm_done = tk.Button(
            master=self.wnd_grid,
            text=f"Click to CONFIRM (only if ALL inferences are done)",
            bg="yellow", fg="black",
            borderwidth=10,
            relief=tk.RAISED,
            command=self.micStt_do_confirm_processing
            )
        self.btn_confirm_done.configure(
            width=( len(self.btn_confirm_done["text"]) + 10 ),
            height=3
            )
        ## configure the grid
        for r_idx in range(self.n_rows):
            self.wnd_grid.rowconfigure(r_idx, weight=1, minsize=20)
            for c_idx in range(self.n_cols):
                self.wnd_grid.columnconfigure(c_idx, weight=1, minsize=20)
        ## populate the 2-D widget array for the grid
        ##       1 - Play and Record Play buttons
        ##       2 to 4 - Labels for Paths of three files
        ##       5 - Buttons for Perform inference
        ##       6 to 8 - Labels for Inference rows
        ##
        ## row 1 - Play and Record buttons
        temp_row_data = []
        r_idx = 0
        for c_idx in range(self.n_cols):
            if c_idx % 2 == 0:
                ## Play button - always enabled
                temp_row_data.append(
                    tk.Button(
                        master=self.wnd_grid,
                        text=f"Play file " + f"{c_idx // 2 + 1} ",
                        bg="yellow", fg="black",
                        borderwidth=5,
                        relief=tk.RAISED,
                        command=partial(
                            self.mic_stt_play_audio_file,
                            c_idx // 2
                            )
                        ))
            else:
                ## Record button - switch decides whether enabled or diabled
                if self.micYorN == 'n':
                    temp_row_data.append(
                        tk.Button(
                            master=self.wnd_grid,
                            text=f"Record file " + f"{c_idx // 2 + 1} ",
                            bg="yellow", fg="black",
                            borderwidth=5,
                            relief=tk.FLAT,
                            state=tk.DISABLED,
                            command=partial(
                                self.mic_stt_rec_audio_file,
                                c_idx // 2
                                )
                            ))
                else:
                    temp_row_data.append(
                        tk.Button(
                            master=self.wnd_grid,
                            text=f"Record file " + f"{c_idx // 2 + 1} ",
                            bg="yellow", fg="black",
                            borderwidth=5,
                            relief=tk.RAISED,
                            state=tk.NORMAL,
                            command=partial(
                                self.mic_stt_rec_audio_file,
                                c_idx // 2
                                )
                            ))
        self.grid_buttons_arr.append(temp_row_data)
        temp_row_data = None
        ## row 2 to 4 - Labels for Paths of three files
        for r_idx in range(1, 4):
            temp_row_data = []
            for c_idx in range(2):
                if c_idx == 0:
                    ## Label for file number
                    temp_row_data.append(
                        tk.Label(
                            master=self.wnd_grid,
                            text="Path file " + f"{r_idx} ",
                            relief=tk.SUNKEN,
                            borderwidth=5
                            ))
                elif c_idx == 1:
                    ## Label for file path 
                    ## no switch check requited as populated appropriately already
                    ##    If no mic - then has 3 hyphens
                    ##    If with mic, then has the actual file paths already
                    temp_row_data.append(
                        tk.Label(
                            master=self.wnd_grid,
                            text=self.wav_files_arr[r_idx-1],
                            relief=tk.SUNKEN,
                            borderwidth=5
                            ))
            self.grid_buttons_arr.append(temp_row_data)
        temp_row_data = None
        ## row 5 - Buttons for Perform inference
        r_idx = 4
        temp_row_data = []
        for c_idx in range(3):
            ## Button to perform STT inference - always enabled
            temp_row_data.append(
                tk.Button(
                    master=self.wnd_grid,
                    text=f"Perform STT for file " + f"{c_idx + 1}",
                    bg="yellow", fg="black",
                    borderwidth=5,
                    relief=tk.RAISED,
                    state=tk.NORMAL,
                    command=partial(
                        self.mic_stt_perform_inference,
                        c_idx
                    )
                    ))
        self.grid_buttons_arr.append(temp_row_data)
        temp_row_data = None
        ## row 6 to 8 - Labels for Inference rows
        for r_idx in range(5, 8):
            temp_row_data = []
            for c_idx in range(2):
                if c_idx == 0:
                    ## Label for Inference number
                    temp_row_data.append(
                        tk.Label(
                            master=self.wnd_grid,
                            text="Inference file " + f"{r_idx - 4}",
                            relief=tk.SUNKEN,
                            borderwidth=5
                            ))
                else:
                    ## Label for STT inference - defaulted to 3 hyphens as inference is still pending
                    temp_row_data.append(
                        tk.Label(
                            master=self.wnd_grid,
                            text=self.ds_infs_arr[r_idx - 5],
                            relief=tk.SUNKEN,
                            borderwidth=5
                            ))
            self.grid_buttons_arr.append(temp_row_data)
        temp_row_data = None
        if self.DEBUG_SWITCH:
            print(f"\nShape of grid_buttons_arr:")
            for idx, row_widgets in enumerate(self.grid_buttons_arr):
                print(f"Row {idx+1} has {len(row_widgets)} widgets")
        ## set grid positions for all the buttons and labels just created
        ## NOTE: going from the button grid 2-d list and mapping to actual grid positions
        for r_idx, row_widgets in enumerate(self.grid_buttons_arr):
            for c_idx, widget in enumerate(row_widgets):
                ## r_idx    c_idx   type
                ## 0        0,2,4   Play button
                ## 0        1,3,5   Record button
                ## 1,2,3    0       Label file number
                ## 1,2,3    1       Label file path
                ## 4        0,1,2   Button to do STT inference
                ## 5,6,7    0       Label Inference number
                ## 5,6,7    1       Label STT Inference value
                if r_idx == 0:
                    ## Button for Play and Record alternating
                    widget.grid(
                        row=0, column=c_idx,
                        rowspan=1, columnspan=1,
                        padx=3, pady=3,
                        sticky="nsew"
                        )
                    widget.configure(
                        width=( len(widget["text"]) + 4 ),
                        height=1
                        )
                elif r_idx in [1, 2, 3]:
                    if c_idx == 0:
                        ## Label file number
                        widget.grid(
                            row=r_idx, column=0,
                            rowspan=1, columnspan=1,
                            padx=3, pady=3,
                            sticky="nsew"
                            )
                        widget.configure(
                            width=( len(widget["text"]) + 4 ),
                            height=3
                            )
                    elif c_idx == 1:
                        ## Label file path
                        widget.grid(
                            row=r_idx, column=1,
                            rowspan=1, columnspan=5,
                            padx=3, pady=3,
                            sticky="nsew"
                            )
                        widget.configure(
                            width=( len(widget["text"]) + 8 ),
                            height=3
                            )
                elif r_idx == 4:
                    ## Button to do STT inference
                    widget.grid(
                        row=4, column=c_idx*2,
                        rowspan=1, columnspan=2,
                        padx=3, pady=3,
                        sticky="nsew"
                        )
                    widget.configure(
                        width=( len(widget["text"]) + 4 ),
                        height=3
                        )
                elif r_idx in [5, 6, 7]:
                    if c_idx == 0:
                        ## Label inference number
                        widget.grid(
                            row=r_idx, column=0,
                            rowspan=1, columnspan=1,
                            padx=3, pady=3,
                            sticky="nsew"
                            )
                        widget.configure(
                            width=( len(widget["text"]) + 4 ),
                            height=5
                            )
                    elif c_idx == 1:
                        ## Label STT inference value output
                        widget.grid(
                            row=r_idx, column=1,
                            rowspan=1, columnspan=5,
                            padx=3, pady=3,
                            sticky="nsew"
                            )
                        widget.configure(
                            width=( len(widget["text"]) + 10 ),
                            height=5
                            )
        ## label for user messages
        r_idx = self.n_rows + 1
        c_idx = 0
        self.lbl_user_msg.grid(
            row=r_idx, column=c_idx,
            rowspan=1, columnspan=self.n_cols + 1,
            sticky="nsew",
            padx=10, pady=10
            )
        ## button for selection confirm
        r_idx = self.n_rows + 2
        c_idx = 0
        self.btn_confirm_done.grid(
            row=r_idx, column=c_idx,
            rowspan=1, columnspan=self.n_cols + 1,
            sticky="nsew",
            padx=10, pady=10
            )
        return
    
    def change_label_user_message_text(self, _in_text, _delay=None):
        self.lbl_user_msg_text.set(_in_text)
        self.wnd_grid.update_idletasks()
        if _delay is not None:
            self.wnd_grid.after(2000)
    
    def mic_stt_play_audio_file(self, _pos):
        ## backup message
        bkup_msg = self.lbl_user_msg["text"]
        ## figure out which button pressed and so which file to play
        file2play = self.wav_files_arr[_pos]
        if file2play != r'---':
            ## change user message
            self.change_label_user_message_text(f"Playing audio {_pos + 1}...", _delay=None)
            myStr = f"\nAttempting to play: {file2play}\n"
            print_and_log(myStr, "info")
            myStr=None
            ## Open the sound file 
            wf = wave.open(file2play, 'rb')
            ## Create an interface to PortAudio
            p = pyaudio.PyAudio()
            # Open a .Stream object to write the WAV file to
            stream = p.open(
                format = p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = wf.getframerate(),
                output = True) ## output = True means play
            ## Play the sound by writing the audio data to the stream while reading in chunks
            data = wf.readframes(self.CHUNCK)            
            while data != b'':
                stream.write(data)
                data = wf.readframes(self.CHUNCK)
            ## wrap up
            stream.close()
            p.terminate()
            myStr = f"\nPlayback successful file {_pos + 1}\n"
            print_and_log(myStr, "info")
            myStr = None
            ## change user message
            self.change_label_user_message_text(f"Playback successful file {_pos + 1}.", _delay=2000)
        else:
            myStr = f"\nPremature Playback attempt...no action taken.\n"
            print_and_log(myStr, "info")
            myStr = None
            ## change user message
            self.change_label_user_message_text(f"Premature Playback attempt...no action taken.", _delay=2000)
        ## restore user message
        self.change_label_user_message_text(bkup_msg, _delay=None)
        return
    
    def mic_stt_rec_audio_file(self, _pos):
        ## backup message
        bkup_msg = self.lbl_user_msg["text"]
        ## figure out which button pressed and so which file to record
        ## first taking from the wav files array coz could be re-recording and so to overwrite it
        file2record = self.wav_files_arr[_pos]
        ## if first time recording then value will be hyphens
        if file2record == r'---':
            ## create name for new file
            file2record = ''.join([
                self.save_wav_here,
                'st_MIC_file',
                str(_pos + 1),
                '.wav'
            ])
        myStr = '\n'.join([
            f"\nRecording parameters:",
            f"chunk                    = {self.CHUNCK}",
            f"channels                 = {self.CHANNELS}",
            f"sampling frequency       = {self.FS} Hz",
            f"sample_format            = 16 bit",
            f"DATA_APPEND_SECS         = {self.DATA_APPEND_SECS}",
            f"Listen time limit (secs) = {self.LISTEN_SECS}",
            f"File to be recorded      =\n\t{file2record}"
        ])
        print_and_log(myStr, "debug")
        myStr = None
        ## actually record and save the file
        ## Create an interface to PortAudio
        p = pyaudio.PyAudio()
        stream = p.open(
            format=self.SAMPLE_FORMAT,
            channels=self.CHANNELS,
            rate=self.FS,
            frames_per_buffer=self.CHUNCK,
            input=True) ## input = True means record
        ## Initialize array to store frames
        frames = []
        start_time = time.time()
        myStr = f"\nStarted listening....\n"
        print_and_log(myStr, "debug")
        myStr = None
        ## change user message
        self.change_label_user_message_text(f"Recording audio {_pos + 1}...", _delay=None)
        while True:
            for i in range(0, int(self.FS / self.CHUNCK * self.DATA_APPEND_SECS)):
                data = stream.read(self.CHUNCK)
                frames.append(data)
                ## notes on what data is:
                ## type(data) = <class 'bytes'>
                ## data =
                ## b'\x00\x00\xfe\xff\x01\x00\x05\x00\xed and continues for a long time
                ## data is a bytes array
            if (time.time() - start_time) > self.LISTEN_SECS:
                myStr = f"\nStopped recording: reached max recording time limit = {self.LISTEN_SECS} seconds"
                print_and_log(myStr, "info")
                myStr = None
                break
        # Stop and close the stream 
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        p.terminate()
        # Save data to WAV file
        wf = wave.open(file2record, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.SAMPLE_FORMAT))
        wf.setframerate(self.FS)
        wf.writeframes(b''.join(frames))
        wf.close()
        ## update in wav file array
        self.wav_files_arr[_pos] = file2record
        ## update the label
        self.grid_buttons_arr[_pos + 1][1]["text"] = file2record
        ## get exact duration from saved wav file
        with contextlib.closing(wave.open(file2record,'r')) as f:
            in_frames = f.getnframes()
            in_rate = f.getframerate()
            rec_secs_exact = in_frames / float(in_rate)
        myStr = '\n'.join([
            f"\nRecorded successfully here: {file2record}",
            f"Duration = {rec_secs_exact:.2f} seconds\n"
        ])
        print_and_log(myStr, "info")
        myStr = None
        ## change user message
        self.change_label_user_message_text(f"Recording successful.", _delay=2000)
        ## restore user message
        self.change_label_user_message_text(bkup_msg, _delay=None)
        return
    
    def replace_certain_words(self, _sent):
        """
        The labels for the objects detected are pre-defined with some words being two words joined.
        E.g. pottedplant , tvmonitor , handbag , diningtable
        Deepspeech will correctly output two words when they are spoken. If they remain two words, the
        Id Key Elements stage will treat them as different words and the match up against predefined
        labels will fail.
        So combine them after the inference is done now as the updated inference.
        """
        replace_map = (
            (r'hand bag', r'handbag'),
            (r'television monitor', r'tvmonitor'),
            (r'dining table', r'diningtable'),
            (r'potted plant', r'pottedplant')
        )
        updated_sent = ''.join(_sent)
        for pair in replace_map:
            rep_this, rep_with = pair
            updated_sent = updated_sent.replace(rep_this, rep_with)
        return updated_sent

    def mic_stt_perform_inference(self, _pos):
        ## backup message
        bkup_msg = self.lbl_user_msg["text"]
        if self.DEBUG_SWITCH:
            print(f"\nDoing INFERERCE FOR FILE {_pos + 1} = {self.wav_files_arr[_pos]}\n")
        file_2_infer = self.wav_files_arr[_pos]
        if file_2_infer == '---':
            myStr = f"\nPremature Inference attempt...no action taken.\n"
            print_and_log(myStr, "info")
            myStr = None
            ## change user message
            self.change_label_user_message_text(f"Premature Inference attempt...no action taken.", _delay=2000)
        else:
            ## change user message
            self.change_label_user_message_text(f"Performing inference for file {_pos + 1}...", _delay=None)
            ## build command and start inferencing
            ds_inf_cmd = self.ds_inf_cmd_fixed + file_2_infer
            myStr = '\n'.join([
                f"\nCommencing STT inference with Deepspeech version 0.7.3.",
                f"on wav file = {file_2_infer}",
                f"\tCommand built as :",
                f"{ds_inf_cmd}"
                ])
            print_and_log(myStr, "info")
            myStr=None
            try:
                inference_run = subprocess.Popen(
                    ds_inf_cmd.split(' '),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True)
                stdout, stderr = inference_run.communicate()
                inference_run.wait()
                inf_output = stdout.rstrip('\n')
                ## take care of words that should be combined for downstream processing to succeed
                updated_inf_output = self.replace_certain_words(inf_output)
                if updated_inf_output != inf_output:
                    myStr = '\n'.join([
                        f"\nWord replacement: CHANGES made",
                        f"Orig inference =\n{inf_output}",
                        f"Changed inference =\n{updated_inf_output}\n"
                        ])
                    print_and_log(myStr, "info")
                    myStr = None
                else:
                    myStr = f"\nWord replacement: NO change\n"
                    print_and_log(myStr, "info")
                    myStr = None
                self.ds_infs_arr[_pos] = updated_inf_output
                self.grid_buttons_arr[_pos + 5][1]["text"] = updated_inf_output
                myStr = f"\nInference output :\n{updated_inf_output}"
                print_and_log(myStr, "info")
                myStr=None
                ## change user message
                self.change_label_user_message_text(f"Inference successful", _delay=2000)
            except Exception as deepspeech_inference_error_msg:
                mystr = '\n'.join([
                    f"\nFATAL ERROR: Problem doing deepspeech inference with this command =\n{ds_inf_cmd}",
                    f"Error message =\n{deepspeech_inference_error_msg}",
                    f"Exiting with Return Code = 30\n"
                    ])
                print_and_log(myStr, "error")
                myStr = None
                ## briefly update user message before exiting
                self.change_label_user_message_text(f"Unexpected problem doing inference. Exiting with Return Code = 30", _delay=3000)
                exit(30)
        ## restore user message
        self.change_label_user_message_text(bkup_msg, _delay=None)
        return

    def micStt_do_confirm_processing(self):
        if self.DEBUG_SWITCH:
            print(f"\nConfirm button pressed\n")
        ## backup message
        bkup_msg = self.lbl_user_msg["text"]
        ## none of the inferences should have the default initial values now. If it happens return without destroyiny
        for inf_val in self.ds_infs_arr:
            if inf_val == "---":
                myStr = f"\nPremature Confirmation attempt...no action taken.\n"
                print_and_log(myStr, "info")
                myStr = None
                ## change user message
                self.change_label_user_message_text(f"Premature Confirmation attempt...no action taken.", _delay=2000)
                ## restore user message
                self.change_label_user_message_text(bkup_msg, _delay=None)
                return
        ## restore user message
        self.change_label_user_message_text(bkup_msg, _delay=None)
        self.root.destroy()
        return

def micStt_show_subwindow(_DEBUG_SWITCH, _root, _micYorN, _save_wav_here, _wavlocfile, _ds_infs_arr, _wav_files_arr):
    if _DEBUG_SWITCH:
        print(f"\nEntered the show grid selection window for wav file processing\n")
    o_micStt_wnd_grid_window = c_micStt_wnd_grid_window(_DEBUG_SWITCH, _root, _micYorN, _save_wav_here, _wavlocfile, _ds_infs_arr, _wav_files_arr)
    o_micStt_wnd_grid_window.wnd_grid.mainloop()
    return

def mic_and_stt_functionality(_micYorN, _save_wav_here, _wavlocfile, _DEBUG_SWITCH):
    '''
    High level module to execute Mic input and Speech to Text.
    Depending on switch, either Mic input logic performed first, or directly goes to STT processing.

    Mic input logic gets user to speak three sentences and saves Wav file for each.
    Wav files saved as per requirements for Deepspeech (16kHz, 16-bit, Mono-channel).
    Will listen for fixed number of seconds.
    User can play each Wav file to listen to it.
    
    Inference performed on the specified Wav files.

    If user unhappy with the recording and/inference, can record that particular sentence again.
    
    VARIABLES PASSED:
          1) Location of folder to save the wav files to.
          2) If running WITH mic input:
             Location where to create and save the text file containing the paths to the newly created wav files.
             OR
             Location of existing file which contains the paths to the already existing wav files.
          3) Debugging switch
    RETURNS:
          1) Return code
                value of 0 => all ok
                non 0      => some problem
          2) Message if return code is non-zero, else will be None
          3) Results: Intermediate data
                      List of paths to wav files just created
    '''
    myStr = f'\n\nRunning {"WITHOUT" if _micYorN == "n" else "WITH"} mic input\n\n'
    print_and_log(myStr, "info")
    myStr = None
    ## array for the abolute paths to the 3 wav files - default running with mic input
    wav_files_arr = ["---", "---", "---"]
    ## if running WITHOUT mic input, read the file and populate now
    if _micYorN == 'n':
        try:
            wav_files_arr = []
            with open(_wavlocfile, 'r') as f:
                for line in f:
                    wav_files_arr.append(line.rstrip("\n"))
            ## check three entries - if problem prematurely return
            if len(wav_files_arr) != 3:
                return_msg = f"\nFATAL ERROR: Expected exactly 3 lines in the file specifying locations of Wav files\n"
                return (110, return_msg, None)
            myStr = f"\nEntries for Wav files:"
            print_and_log(myStr, "info")
            myStr = None
            for idx, each_wav_file in enumerate(wav_files_arr):
                if not os.path.isfile(each_wav_file):
                    return_msg = f"\nFATAL ERROR: Check the entries for Wav files in input file.\nFile does not exist:\n{each_wav_file}\n"
                    return (115, return_msg, None)
                else:
                    myStr = f"\t{idx+1}) {each_wav_file}"
                    print_and_log(myStr, "info")
                    myStr = None
        except Exception as wavlocfile_processing_error:
            return_msg = f"\nFATAL ERROR: Problem reading file of Wav locations.\nError message: {wavlocfile_processing_error}n"
            return (100, return_msg, None)
    ## array to hold final results - will be the module results passed to next stage
    ##       three entries of string for each of the wav files processed by STT
    ## initialise as hyphens - to be overwritten later
    deepspeech_inferences_arr = ["---", "---", "---"]
    ## make a root window and show it
    o_micSttRootWindow = c_micSttRootWindow(_DEBUG_SWITCH, _micYorN, _save_wav_here, _wavlocfile, deepspeech_inferences_arr, wav_files_arr)
    o_micSttRootWindow.root.mainloop()
    ## save required stuff
    ## saving of wav files to directory already done at this point
    ## create the file for the saved wavs on if this is with mic input run
    if _micYorN == 'y':
        if "---" not in wav_files_arr:
            try:
                data_2_write = ""
                with open(_wavlocfile, "w") as f:
                    data_2_write = "\n".join(wav_files_arr) + "\n"
                    f.write(data_2_write)
                myStr = f"\nFile for wavlocfile created containing locations of Wav files\n"
                print_and_log(myStr, "info")
                myStr = None
            except Exception as wavlocfile_save_error:
                return_msg = f"\nFATAL ERROR: Problem saving data to the Wav locations file.\nError message: {wavlocfile_save_error}\n"
                return (150, return_msg, None)
        else:
            ## invalid entries in the wav files array - do not create the file
            return_msg = f"\nFATAL ERROR: Did not create the Wav locations file as entries of HYPHENS found\n"
            return(160, return_msg, None)
    return (0, None, deepspeech_inferences_arr)

###############################################################################################################
## -------------------------  MIC + STT LOGIC ENDS              MIC + STT LOGIC  ENDS ------------------------- 
###############################################################################################################

## ----------------------------------------------------------------------------------------------------------------------

###############################################################################################################
## ---------------------------  KEYWORDS LOGIC STARTS        KEYWORDS LOGIC  STARTS --------------------------- 
###############################################################################################################

class c_idkeyelem_wnd_grid_window:

    def __init__(self, _root, _sentence_num, _each_sentence_candidate_keywords, _index_positions_deselected):
        self.root = _root
        self.sentence_num = _sentence_num
        self.each_sentence_candidate_keywords = _each_sentence_candidate_keywords
        self.index_positions_deselected = _index_positions_deselected
        
        ## array for Buttons - self.buttons_arr_for_words
        ##    later will be populated to become a list of lists
        ##    each inner list will be for a row,
        ##    each element in inner list will be entry for that column in the row
        ##    thus the overall array become a 2 dimensional array of 6 rows x 5 columns
        ##         that can hold up to 30 candidate keywords
        ##    each element will be one label to hold one keyword
        ##
        ##    Button characteristics indicates if its, got no word, or has a Selected word, or a Deselected word
        ##        For a button with a valid word in it - STATE is Normal
        ##        Situation Description        Relief        Borderwidth      Color bg          Color fg         State
        ##        1) Valid word, button        SUNKEN            10            green             white           Normal
        ##             is SELECTED
        ##        2) Valid word, button        RAISED            10            yellow            black           Normal
        ##             is DESELCTED
        ##        3) No word in                FLAT              4             black             white           Disabled
        ##             input data
        self.buttons_arr_for_words = []
        
        self.n_rows_words = 6
        self.n_cols_words = 5
        ## number of words that can be selected for confirmation processing
        self.valid_selection_counts = [0, 1, 2, 3]
        self.keywords_selected_count_at_start = len(self.each_sentence_candidate_keywords)
        self.keywords_selected_count_current = self.keywords_selected_count_at_start
        
        self.wnd_grid = tk.Toplevel(master=self.root)
        self.wnd_grid.title(f"Selection Window for Keywords -- Sentence number {self.sentence_num}")
        ## label for count of current selections
        self.lbl_track_selected_count = tk.Label(
            master=self.wnd_grid,
            text=" ".join( [ "Count of Words currently Selected = ", str(self.keywords_selected_count_at_start) ]),
            relief=tk.FLAT,
            borderwidth=10
            )
        self.lbl_track_selected_count.configure(
            width= ( len(self.lbl_track_selected_count["text"]) + 10 ),
            height=5
            )
        if self.keywords_selected_count_at_start not in self.valid_selection_counts:
            self.lbl_track_selected_count.configure(bg="red", fg="white")
        else:
            self.lbl_track_selected_count.configure(bg="green", fg="white")
        ## button for selection confirm
        self.btn_confirm_selections = tk.Button(
            master=self.wnd_grid,
            text="Click to Confirm Deselections",
            bg="yellow", fg="black",
            borderwidth=10,
            relief=tk.RAISED,
            command=self.do_confirm_selections_processing
            )
        self.btn_confirm_selections.configure(
            width= ( len(self.btn_confirm_selections["text"]) + 20 ),
            height=5
            )
        if self.keywords_selected_count_current not in self.valid_selection_counts:
            self.btn_confirm_selections.configure(state=tk.DISABLED, relief=tk.FLAT)
        
        ## populate the button array for the grid
        ##    first make skeleton entries for the buttons
        ##    by default assume no word is present to display so all buttons are in disabled state
        ##       with the text saying "No Data"
        for r_idx in range(self.n_rows_words):
            self.wnd_grid.columnconfigure(r_idx, weight=1, minsize=50)
            self.wnd_grid.rowconfigure(r_idx, weight=1, minsize=50)
            temp_row_data = []
            for c_idx in range(self.n_cols_words):
                temp_row_data.append(
                    tk.Button(
                        master=self.wnd_grid,
                        text="No Data",
                        bg="black", fg="white",
                        relief=tk.FLAT,
                        borderwidth=10,
                        state=tk.DISABLED
                        )
                    )
            self.buttons_arr_for_words.append(temp_row_data)
        
        ## now populate the words and activate where applicable
        idx = 0 ## index to access each keyword from the input list
        for r_idx in range(self.n_rows_words):
            for c_idx in range(self.n_cols_words):
                ## set grid position for all the label elements
                self.buttons_arr_for_words[r_idx][c_idx].grid(
                    row=r_idx, column=c_idx,
                    padx=5, pady=5,
                    sticky="nsew"
                    )
                ## check if word is available from the input array
                if idx < self.keywords_selected_count_at_start:
                    ## Yes, then put the word as the text and by default make the button as Selected
                    self.buttons_arr_for_words[r_idx][c_idx].configure(
                        text=self.each_sentence_candidate_keywords[idx],
                        bg="green", fg="white",
                        relief=tk.SUNKEN,
                        state=tk.NORMAL,
                        command=partial(
                            self.word_button_clicked,
                            r_idx, c_idx
                            )
                        )
                    idx += 1
        
        ## label for Current Count of Selected buttons
        r_idx = self.n_rows_words
        c_idx = 0
        self.lbl_track_selected_count.grid(
            row=r_idx, column=c_idx,
            rowspan=1, columnspan=self.n_cols_words,
            sticky="nsew",
            padx=5, pady=5
            )
        ## label for Confirm Selections
        r_idx = self.n_rows_words + 1
        c_idx = 0
        self.btn_confirm_selections.grid(
            row=r_idx, column=c_idx,
            rowspan=1, columnspan=self.n_cols_words,
            sticky="nsew",
            padx=5, pady=5
            )
    
    def word_button_clicked(self, _r_idx, _c_idx):

        ## change button characteristics to indicate toggle of Select / Deselect of word
        if self.buttons_arr_for_words[_r_idx][_c_idx]["bg"]=="green":
            ## button is currently Selected, change to Deselected characteristics
            self.buttons_arr_for_words[_r_idx][_c_idx].configure(
                relief=tk.RAISED,
                bg="yellow", fg="black"
                )
            self.keywords_selected_count_current -= 1
        else:
            ## button is currently Deselected, change to Selected characteristics
            self.buttons_arr_for_words[_r_idx][_c_idx].configure(
                relief=tk.SUNKEN,
                bg="green", fg="white"
                )
            self.keywords_selected_count_current += 1
        
        ## update the label for count of active selections
        self.lbl_track_selected_count.configure(
            text=" ".join( [ "Count of Words currently Selected = ", str(self.keywords_selected_count_current) ]),
            )
        self.lbl_track_selected_count.configure(
            width= ( len(self.lbl_track_selected_count["text"]) + 10 ),
            height=5
            )
        ## depending on current count of selections, change the color of the label
        ##    also, disable the button for confirm selections  if required
        if self.keywords_selected_count_current not in self.valid_selection_counts:
            self.lbl_track_selected_count.configure(bg="red", fg="white")
            self.btn_confirm_selections.configure(state=tk.DISABLED, relief=tk.FLAT)
        else:
            self.lbl_track_selected_count.configure(bg="green", fg="white")
            self.btn_confirm_selections.configure(state=tk.NORMAL, relief=tk.RAISED)
        ## make the Confirm Selections button Disabled if the number of selections is not valid
    
    def do_confirm_selections_processing(self):
        myStr = f"\n\nCONFIRM SELECTIONS BUTTON PRESSED\n"
        print_and_log(myStr, "debug")
        myStr = None
        
        ## self.lbl_root_error_message   do something with this later - maybe not required as am disabling
        ##        confirm button if current count is not valid to proceed

        ## For the Deselected keywords, figure out the position and add position number
        ##     to the return list for action later.
        #self.index_positions_deselected
        for r_idx in range(self.n_rows_words):
            for c_idx in range(self.n_cols_words):
                if self.buttons_arr_for_words[r_idx][c_idx]["bg"] == "yellow":
                    self.index_positions_deselected.append( r_idx * self.n_cols_words + c_idx )
        
        myStr = f"\nDeselected positions=\n{self.index_positions_deselected}\n\n"
        print_and_log(myStr, "debug")
        myStr = None
        self.root.destroy()

def generic_show_grid_selections_window(_root, _sentence_num, _each_sentence_candidate_keywords, _index_positions_deselected):
    o_idkeyelem_wnd_grid_window = c_idkeyelem_wnd_grid_window(_root, _sentence_num, _each_sentence_candidate_keywords, _index_positions_deselected)
    o_idkeyelem_wnd_grid_window.wnd_grid.mainloop()

class c_idkeyelem_root_window:

    def __init__(self, _DEBUG_SWITCH, _sentence_num, _each_sentence_candidate_keywords, _index_positions_deselected):
        self.sentence_num = _sentence_num
        self.each_sentence_candidate_keywords = _each_sentence_candidate_keywords
        ## this is the list passed from beginning. to be populated with positions of deselections during the
        ##      grid window processing
        self.index_positions_deselected = _index_positions_deselected

        self.msg_warning_for_root_do_not_close = r'DO NOT CLOSE THIS WINDOW TILL ALL SELECTIONS ARE COMPLETED FOR THIS SENTENCES KEYWORDS!!!'
        self.msg_instructions_for_root = "".join([
            f"                               ----------------------------------",
            f"\n                               -----      INSTRUCTIONS      -----",
            f"\n                               ----------------------------------",
            f"\n",
            f"\n",
            f"\nThis is the main control window for keywords selection for the sentence being processed.",
            f"\n",
            f"\n\t1.  Click on the button below to proceed.",
            f"\n",
            f"\n\t2.  A grid will display showing the candidate keywords as CLICKABLE BUTTONS.",
            f"\n\t         At start, ALL the words will be selected."
            f"\n\t         Important: If there are no words to display, you cannot click to proceed to"
            f"\n\t                    the grid selection window. Simply close this window and the"
            f"\n\t                    selection process for next sentence will begin."
            f"\n",
            f"\n\t3.  You can Select or Deselect a Keyword by clicking the button for the word.",
            f"\n",
            f"\n\t4.  Once ready, click the button to Confirm Deselections.",
            f"\n\t         NOTE; You can only Select from 0 to 3 keywords.",
            f"\n\t         Monitor the count of current Selections before confirming Deselections.",
            f"\n",
            f"\n\t5.  Important: If you accidentally close this window, the selection process for next",
            f"\n\t               sentence will start automatically.",
            f"\n"
        ])

        ## root window for the sentence being processed
        self.root = tk.Tk()
        if len(self.each_sentence_candidate_keywords) == 0:
            self.root.title(f"Word selection - Sentence number {_sentence_num} - No Words to Display -- Please close this window")
        else:
            self.root.title(f"Word selection - Sentence number {_sentence_num} - Click button to Proceed")
        ## label for warning message not to close
        self.lbl_root_msg_warning_not_close = tk.Label(
            master=self.root,
            text=self.msg_warning_for_root_do_not_close,
            bg="red", fg="white",
            width=(len(self.msg_warning_for_root_do_not_close) + 10),
            height=5
            )
        ## label for instructions
        self.lbl_root_instructions = tk.Label(
            master=self.root,
            text=self.msg_instructions_for_root,
            bg="blue", fg="white",
            justify=tk.LEFT,
            width=130,
            height=25,
            relief=tk.SUNKEN
            )
        ## button to proceed to grid selection window
        ## assume there are words and make proceed button clickable
        self.btn_root_click_proceed = tk.Button(
            master=self.root,
            text=f"Click to proceed to selection - Sentence number {_sentence_num}",
            bg="yellow", fg="black",
            relief=tk.RAISED,
            borderwidth=10,
            command=partial(
                generic_show_grid_selections_window,
                self.root,
                self.sentence_num,
                self.each_sentence_candidate_keywords,
                self.index_positions_deselected
                )
            )
        ## if no words to display in grid, disable the button to proceed and change the text displayed in button
        if len(self.each_sentence_candidate_keywords) == 0:
            self.btn_root_click_proceed.configure(
            state=tk.DISABLED,
            relief=tk.FLAT,
            text=f"No words to make Selections for Sentence number {_sentence_num} - Please close this window",
            )
        self.btn_root_click_proceed.configure(
            width=100,
            height=7
        )
        self.lbl_root_instructions.pack(padx=10, pady=10)
        self.lbl_root_msg_warning_not_close.pack(padx=15, pady=15)
        self.btn_root_click_proceed.pack(padx=50, pady=50)

def change_candidate_elements(_DEBUG_SWITCH, _sentence_num, _each_sentence_candidate_keywords, _index_positions_deselected):
    '''
    ACTIONS:
        Shows the keywords list for each sentence before any changes are made.
        Accepts user selection for the word positions to be dropped. Validates the values entered.
        Drops the words specified by user selection if allowed to.
        Calls the sanity check function to ensure the remaining words meet the requirements for
              query module in downstream processing.
    ACCEPTS:
        1) Debug switch
        2) Sentence number starting from 1
        3) List of keywords for particular sentence
        4) Array to fill with the deselected positions (starting from 0 index position)
    RETURN:
        1) changes_made_flag : A boolean value indicating if any changes made
                False if no changes required and/ or made by user
                True  if any changes are made
        2) the keywords list (changed or unchanged as per user selection)
    '''
    ## show keywords before any changes
    myStr = "\n".join([
        f"\n\nCandidate key words BEFORE any changes for sentence {_sentence_num} :",
        f"{_each_sentence_candidate_keywords}\n"
        ])
    print_and_log(myStr, "debug")
    myStr = None
    
    ## create and show the root window
    o_idkeyelem_root_window = c_idkeyelem_root_window(_DEBUG_SWITCH, _sentence_num, _each_sentence_candidate_keywords, _index_positions_deselected)
    o_idkeyelem_root_window.root.mainloop()

def id_elements_functionality(_stt_module_results, _opfileallposinfo, _DEBUG_SWITCH):
    '''
    High level module to execute the identify keywords functionality by processing the 
         transcriptions from the STT functionality.
    VARIABLES PASSED:
          1) Array containing the transcriptions i.e. output of the STT module
          2) Location of file where to write the POS info for all the words that are not stopwords
          3) Debugging switch
    RETURNS:
          1) Return code
                value of 0 => all ok
                non 0      => some problem
          2) Message if return code is non-zero, else will be None
          3) Results array - containing the keywords in the required data structure
    '''

    ## We want to create a set of candidate keywords to present to the user as part of gui selection
    ## Start with the transcription sentences from STT block output.
    ##       process to do folllowing for each sentence:
    ##            1) Word tokenize
    ##            2) Drop all stop-words
    ##            3) Do POS tagging, only keep words that are part of the valid type of POS
    ##            4) Take the lemma form of the word, not the original word itself
    ##            5) Only retain word if it appears in the set of object label class names
    ##               It is pointless to present words to the user for selection that are never
    ##               going to return hits in the database query.
    ## 
    ## Now we have the candidate keywords for the setnence for gui selection logic

    ## setup spacy
    nlp = spacy.load('en_core_web_lg')
    POS_INTEREST_INCLUDE_VERBS_SWITCH = False
    
    ## from the input data, if present, remove characters for newline and period
    sentences_orig = [] # sentences to process
    for each_transcription in _stt_module_results:
        sentences_orig.append(each_transcription.rstrip("\n").rstrip("."))
    
    ## create various arrays required
    ## nlp documents array - one per original sentence
    docs = [nlp(each_sentence) for each_sentence in sentences_orig]
    ## array for tokens of the original sentences - create list of word tokens for each sentence doc
    sentences_words_list = [[token.text for token in doc] for doc in docs]
    ## create list of words for each sentence doc - WITHOUT STOP WORDS
    sentences_words_list_no_stop = [[word for word in words_list if not nlp.vocab[word].is_stop ] for words_list in sentences_words_list]
    ## sentences array with stop words removed - only used to display for readability
    sentences_no_stop = [' '.join(words_list) for words_list in sentences_words_list_no_stop]
    
    myStr = f"\n\nThe following sentences will be processed:\n"
    print_and_log(myStr, "debug")
    myStr = None
    for idx, each_input_sentence in enumerate(sentences_orig):
        myStr = f"\tSentence {idx+1} :\n{each_input_sentence}"
        print_and_log(myStr, "debug")
        myStr = None
    
    myStr = f"\n\nWords of each input sentence:\n"
    print_and_log(myStr, "info")
    myStr = None
    for idx, words_list in enumerate(sentences_words_list):
        myStr = f"\tSentence {idx+1} :\n{words_list}"
        print_and_log(myStr, "info")
        myStr = None
    
    myStr = f"\n\nWords of each input sentence after removing all stop words:\n"
    print_and_log(myStr, "info")
    myStr = None
    for idx, words_list in enumerate(sentences_words_list_no_stop):
        myStr = f"\tSentence {idx+1} :\n{words_list}"
        print_and_log(myStr, "info")
        myStr = None
    
    myStr = f"\n\nJoining the non-stop words as a new sentence (for readability only):\n"
    print_and_log(myStr, "info")
    myStr = None
    for idx, new_sent_no_stop in enumerate(sentences_no_stop):
        myStr = f"\tNew sentence {idx+1} :\n{new_sent_no_stop}"
        print_and_log(myStr, "info")
        myStr = None
    
    ## pos extraction and fill data structure
    pos_info = [[{} for seach_word in words_list] for words_list in sentences_words_list_no_stop]
    for idx1, each_sent_no_stop in enumerate(sentences_no_stop):
        doc = nlp(each_sent_no_stop)
        for idx2, token in enumerate(doc):
            pos_info[idx1][idx2]['text']     = token.text
            pos_info[idx1][idx2]['lemma_']   = token.lemma_
            pos_info[idx1][idx2]['pos_']     = token.pos_
            pos_info[idx1][idx2]['tag_']     = token.tag_
            pos_info[idx1][idx2]['dep_']     = token.dep_
            pos_info[idx1][idx2]['shape_']   = token.shape_
            pos_info[idx1][idx2]['is_alpha'] = token.is_alpha
            pos_info[idx1][idx2]['is_stop']  = token.is_stop
    
    myStr = f"\n\nAll non-stop words pos info:\n"
    print_and_log(myStr, "info")
    myStr = None
    for idx1, each_new_sent in enumerate(pos_info):
        myStr = f"........."
        print_and_log(myStr, "info")
        myStr = None
        for idx2, each_word_pos_info in enumerate(each_new_sent):
            myStr = f"Sentence {idx1+1}, word {idx2+1} :\n{each_word_pos_info}"
            print_and_log(myStr, "info")
            myStr = None

    ## save the pos info to file for all words that are not stopwords
    try:
        with open(_opfileallposinfo, 'w') as opfileall:
                json.dump(pos_info, opfileall)
        myStr = f"\n\nAll POS info file successfully created here:\n{_opfileallposinfo}\n\n"
        print_and_log(myStr, "info")
        myStr = None
    except Exception as opfileallpos_write_error:
        return_msg = f"\nFATAL ERROR: Problem creating the output file for all words pos info.\nError message: {opfileallpos_write_error}\nRC=200"
        return (200, return_msg, None)
    
    ## extract lemma form of each word that matches a tag of interest.
    ##     note that only Nouns will be considered by default, unless the switch for verbs is True.
    ## tags of interest as per naming convention here: https://spacy.io/api/annotation#pos-tagging
    pos_tags_of_interest_nouns = ['NN', 'NNS', 'NNP']
    pos_tags_of_interest_verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    if POS_INTEREST_INCLUDE_VERBS_SWITCH:
        pos_tags_of_interest = pos_tags_of_interest_nouns + pos_tags_of_interest_verbs
    else:
        pos_tags_of_interest = pos_tags_of_interest_nouns
    
    ## object classes that object detector is trained on
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
            "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
            "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    labels_set = set(labels)

    ## array for candidate keywords
    candidate_keywords = []
    for sent_info in pos_info:
        each_sentence_candidate_keywords = []
        for each_word_info in sent_info:
            if each_word_info['tag_'] in pos_tags_of_interest:
                each_sentence_candidate_keywords.append(each_word_info['lemma_'])
        each_sentence_candidate_keywords_set = set(each_sentence_candidate_keywords)
        ## e.g.   {'story', 'fruit', 'banana'} - ( {'story', 'fruit', 'banana'} - {'apple', 'pear', 'banana'} )   =
        ##        {'story', 'fruit', 'banana'} - {'story', 'fruit'}                                               =
        ##        {'banana'}   --> which is what we need finally
        each_sentence_candidate_keywords_labels_matched = list( each_sentence_candidate_keywords_set - (each_sentence_candidate_keywords_set - labels_set) )
        if each_sentence_candidate_keywords_labels_matched: # append to the final list only if its not an empty list
            candidate_keywords.append(each_sentence_candidate_keywords_labels_matched)
    
    myStr = f"\n\nCandidate keywords AFTER matching against class labels:\n{candidate_keywords}\n\n"
    print_and_log(myStr, "info")
    myStr = None

    ## cleanup for gc
    del labels_set, labels
    
    ## execute gui logic to allow deselection of keywords, check the number of selections are valid,
    ##         capture the deselected positions
    index_positions_deselected_all = []
    for sentence_num, each_sentence_candidate_keywords in enumerate(candidate_keywords):
        index_positions_deselected = []
        change_candidate_elements(
            _DEBUG_SWITCH,
            sentence_num + 1,
            each_sentence_candidate_keywords,
            index_positions_deselected
            )
        index_positions_deselected_all.append(index_positions_deselected)
    
    ## extract the Selected keywords into new data structure
    final_keywords = []
    for each_index_positions_deselected, each_sentence_candidate_keywords in \
        zip(index_positions_deselected_all, candidate_keywords):
        temp_arr = [candidate_keyword for idx, candidate_keyword in enumerate(each_sentence_candidate_keywords) if idx not in each_index_positions_deselected]
        final_keywords.append(temp_arr)
    
    for idx, (each_candidate_keywords, each_index_positions_deselected, each_final_keywords) in \
        enumerate(zip(candidate_keywords, index_positions_deselected_all, final_keywords)):
        myStr = "\n".join([
            f"\nSentence {idx+1}:",
            f"BEFORE = {each_candidate_keywords}",
            f"Deselected = {each_index_positions_deselected}",
            f"AFTER = {each_final_keywords}\n"
            ])
        print_and_log(myStr, "debug")
        myStr = None
    
    return (0, None, final_keywords)

###############################################################################################################
## -----------------------------  KEYWORDS LOGIC ENDS        KEYWORDS LOGIC  ENDS ----------------------------- 
###############################################################################################################

## ----------------------------------------------------------------------------------------------------------------------

###############################################################################################################
## ----------------------  NEO4J QUERY LOGIC STARTS             NEO4J QUERY LOGIC STARTS ---------------------- 
###############################################################################################################

def check_input_data(_key_elements_list):
    '''
    Perform santiy check on the input data.
    1) Check it is a list of lists.
    2) Outer list must contain exactly 2 or 3 elements.
    3) Each inner list to consist of 1 or 2 or 3 elements.
    4) All inner list elements to be strings.
    VARIABLES PASSED:
          1) list containing the objects to find within individual image
    RETURNS:
          1) return code = 0 means all ok, non-zero means problem
            value       error situation
            100         main data structure is not a list
            105         outer list did not contain exacty 1/2/3 elements
            110         some inner element is not a list
            115         some inner list did not contain exactly 0/1/2/3 elements
            120         some element of the inner list is not a string
    '''
    valid_count_words_in_inner_list = [0, 1, 2, 3]
    valid_length_of_outer_list = [1, 2, 3]
    if type(_key_elements_list) != list:
        myStr = f"\n\nERROR 100: Expected a list of lists as the input data. Outer data is not a list. Please check the input and try again.\n"
        print_and_log(myStr, 'error')
        myStr = None
        return 100
    elif len(_key_elements_list) not in valid_length_of_outer_list:
        myStr = f"\n\nERROR 105: Expected a list with exactly 1/ 2/ 3 elements. Please check the input and try again.\n"
        print_and_log(myStr, 'error')
        myStr = None
        return 105
    for inner_list in _key_elements_list:
        if type(inner_list) != list:
            myStr = f"\n\nERROR 110: Expected a list of lists as the input data. Inner data is not a list. Please check the input and try again.\n"
            print_and_log(myStr, 'error')
            myStr = None
            return 110
        elif len(inner_list) not in valid_count_words_in_inner_list:
            myStr = f"\n\nERROR 115: Some inner list did not contain expected number of elements i.e. from 0 to 3. Please check the input and try again.\n"
            print_and_log(myStr, 'error')
            myStr = None
            return 115
        for object_string in inner_list:
            if type(object_string) != str:
                print()
                myStr = f"\n\nERROR 120: Expected a list of lists as the input data, with inner list consisting of strings. Please check the input and try again.\n"
                print_and_log(myStr, 'error')
                myStr = None
                return 120
    return 0 ## all ok

def query_neo4j_db(_each_inner_list, _in_limit):
    '''
    Queries database and returns the distinct results.
    VARIABLES PASSED:
          1) list containing the objects to find within individual image
          2) limit value for number of results from neo4j
    RETURNS:
          tuple of (return code, results as list of dictionarary items)
    Return code = 0 means all ok
            value       error situation
            100         unable to connect to database
            1000        unexpected situation
            1500        problem querying the neo4j database
    '''
    ## set result as None by default
    result = None
    
    ## establish connection to Neo4j
    try:
        graph = Graph(uri="bolt://localhost:7687",auth=("neo4j","abc"))
    except Exception as error_msg_neo_connect:
        myStr = "\n".join([
            f"\nUnexpected ERROR connecting to neo4j. Function call return with RC=100. Message:",
            f"{error_msg_neo_connect}\n\n"
            ])
        print_and_log(myStr, 'error')
        myStr = None
        # print(f"\nUnexpected ERROR connecting to neo4j. Function call return with RC=100. Message:\n{error_msg_neo_connect}\n\n")
        return (100, result)
    
    ## build the query and execute
    """
    ## version without any HAS reln score threshold and without source dataset restrictions - START
    ## version without any HAS reln score threshold and without source dataset restrictions
    stmt1_three_objects = r'MATCH (o1:Object)--(i:Image)--(o2:Object)--(i)--(o3:Object) ' + \
        r'WHERE o1.name = $in_obj1 AND o2.name = $in_obj2 AND o3.name = $in_onj3 ' + \
        r'RETURN DISTINCT i.name as Image, i.dataset as Source ' + \
        r'LIMIT $in_limit'
    stmt2_two_objects = r'MATCH (o1:Object)--(i:Image)--(o2:Object) ' + \
        r'WHERE o1.name = $in_obj1 AND o2.name = $in_obj2 ' + \
        r'RETURN DISTINCT i.name as Image, i.dataset as Source ' + \
        r'LIMIT $in_limit'
    stmt3_one_objects = r'MATCH (o1:Object)--(i:Image) ' + \
        r'WHERE o1.name = $in_obj1 ' + \
        r'RETURN DISTINCT i.name as Image, i.dataset as Source ' + \
        r'LIMIT $in_limit'
    ## version without any HAS reln score threshold and without source dataset restrictions - END
    """
    ## HAS reln score > 90.0 and source dataset either flickr30k or coco_2017_test - START
    stmt1_three_objects = r'MATCH (o1:Object)-[r1:HAS]-(i:Image)-[r2:HAS]-(o2:Object)--(i)-[r3:HAS]-(o3:Object) ' + \
        r'WHERE o1.name = $in_obj1 AND o2.name = $in_obj2 AND o3.name = $in_onj3 ' + \
        r'AND r1.score > $in_r1_score AND r2.score > $in_r2_score AND r3.score > $in_r3_score ' + \
        r'AND i.dataset IN [$in_src_ds_1 , $in_src_ds_2]' + \
        r'RETURN DISTINCT i.name as Image, i.dataset as Source ' + \
        r'LIMIT $in_limit'
    stmt2_two_objects = r'MATCH (o1:Object)-[r1:HAS]-(i:Image)-[r2:HAS]-(o2:Object) ' + \
        r'WHERE o1.name = $in_obj1 AND o2.name = $in_obj2 ' + \
        r'AND r1.score > $in_r1_score AND r2.score > $in_r2_score ' + \
        r'AND i.dataset IN [$in_src_ds_1 , $in_src_ds_2]' + \
        r'RETURN DISTINCT i.name as Image, i.dataset as Source ' + \
        r'LIMIT $in_limit'
    stmt3_one_objects = r'MATCH (o1:Object)-[r1:HAS]-(i:Image) ' + \
        r'WHERE o1.name = $in_obj1 ' + \
        r'AND r1.score > $in_r1_score ' + \
        r'AND i.dataset IN [$in_src_ds_1 , $in_src_ds_2]' + \
        r'RETURN DISTINCT i.name as Image, i.dataset as Source ' + \
        r'LIMIT $in_limit'
    ## HAS reln score > 90.0 and source dataset either flickr30k or coco_2017_test - END
    try:
        tx = graph.begin()
        """
        ## version without any HAS reln score threshold and without source dataset restrictions - START
        if len(_each_inner_list) == 3:
            ## query to find images containing all three objects passed
            in_obj1, in_obj2, in_onj3 = _each_inner_list
            result = tx.run(stmt1_three_objects, parameters={'in_obj1': in_obj1, 'in_obj2': in_obj2, 'in_onj3': in_onj3, 'in_limit': _in_limit}).data()
        elif len(_each_inner_list) == 2:
            ## query to find images containing all two objects passed
            in_obj1, in_obj2 = _each_inner_list
            result = tx.run(stmt2_two_objects, parameters={'in_obj1': in_obj1, 'in_obj2': in_obj2, 'in_limit': _in_limit}).data()
        elif len(_each_inner_list) == 1:
            ## query to find images containing all one objects passed
            in_obj1 = _each_inner_list[0]
            result = tx.run(stmt3_one_objects, parameters={'in_obj1': in_obj1, 'in_limit': _in_limit}).data()
        ## version without any HAS reln score threshold and without source dataset restrictions - END
        """
        ## HAS reln score > 90.0 and source dataset either flickr30k or coco_2017_test - START
        if len(_each_inner_list) == 3:
            ## query to find images containing all three objects passed
            in_obj1, in_obj2, in_onj3 = _each_inner_list
            #result = tx.run(stmt1_three_objects, parameters={'in_obj1': in_obj1, 'in_obj2': in_obj2, 'in_onj3': in_onj3, 'in_limit': _in_limit}).data()
            result = tx.run(stmt1_three_objects, parameters={
                'in_obj1': in_obj1, 'in_obj2': in_obj2, 'in_onj3': in_onj3, 'in_limit': _in_limit, 'in_r1_score': 90.0, 'in_r2_score': 90.0, 'in_r3_score': 90.0, 'in_src_ds_1': 'flickr30k', 'in_src_ds_2': 'coco_test_2017'
                }).data()
        elif len(_each_inner_list) == 2:
            ## query to find images containing all two objects passed
            in_obj1, in_obj2 = _each_inner_list
            #result = tx.run(stmt2_two_objects, parameters={'in_obj1': in_obj1, 'in_obj2': in_obj2, 'in_limit': _in_limit}).data()
            result = tx.run(stmt2_two_objects, parameters={
                'in_obj1': in_obj1, 'in_obj2': in_obj2, 'in_limit': _in_limit, 'in_r1_score': 90.0, 'in_r2_score': 90.0, 'in_src_ds_1': 'flickr30k', 'in_src_ds_2': 'coco_test_2017'
                }).data()
        elif len(_each_inner_list) == 1:
            ## query to find images containing all one objects passed
            in_obj1 = _each_inner_list[0]
            #result = tx.run(stmt3_one_objects, parameters={'in_obj1': in_obj1, 'in_limit': _in_limit}).data()
            result = tx.run(stmt3_one_objects, parameters={
                'in_obj1': in_obj1, 'in_limit': _in_limit, 'in_r1_score': 90.0, 'in_src_ds_1': 'flickr30k', 'in_src_ds_2': 'coco_test_2017'
                }).data()
        ## HAS reln score > 90.0 and source dataset either flickr30k or coco_2017_test - END
        elif len(_each_inner_list) == 0: 
            ## possible because user deselected all the keywords for a particular sentence
            result = []
        else:
            myStr = "\n".join([
                f"\n\nUnexpected length of the key elements array. Result set to None.",
                f"Function call return with RC=1000.\n\n"
                ])
            print_and_log(myStr, 'error')
            myStr = None
            result = None
            return (1000, result)
        #tx.commit()
        #while not tx.finished():
        #    pass # tx.finished return True if the commit is complete
    except Exception as error_msg_neo_read:
        myStr = "\n".join([
            f"\n\nUnexpected ERROR querying neo4j.",
            f"Message:\n{error_msg_neo_read}",
            f"Function call return with RC=1500.\n\n"
            ])
        print_and_log(myStr, 'error')
        myStr = None
        return (1500, result)
    ## return tuple of return code and the results. RC = 0 means no errors.
    return (0, result)

def query_neo4j_functionality(_id_elements_module_results ,_query_result_limit, _DEBUG_SWITCH):
    '''
    High level module to execute the querying of database to fetch images based on the keywords identified.
    VARIABLES PASSED:
          1) Array containing the keywords identified as an expected data structure
          2) Value to limit the query result to
          3) Debugging switch
    RETURNS:
          1) Return code
                value of 0 => all ok
                non 0      => some problem
          2) Message if return code is non-zero, else will be None
          3) Results array - the database query info
    ''' 
    key_elements_list = copy.deepcopy(_id_elements_module_results)
    
    myStr = f"\n\ntype key_elements_list = {type(key_elements_list)}\nkey_elements_list =\n{key_elements_list}"
    print_and_log(myStr, "debug")
    myStr = None

    ## perform sanity checks on the keywords data passed
    if check_input_data(key_elements_list):
        # print(f"\nFATAL ERROR: Problem with the data input to neo4j query function. Exiting with return code 300.")
        # exit(300)
        return_msg = f"\nFATAL ERROR: Problem with the data input to neo4j query function. RC=300."
        return (300, return_msg, None)
    
    ## Note: each query to db can return None if there are no words in the query
    query_results_arr = []
    ## query the database with the input data
    #_query_result_limit = 10
    for each_inner_list in key_elements_list:
        query_rc, query_result = query_neo4j_db(each_inner_list, _query_result_limit)
        if query_rc != 0:
            # print(f"\nFATAL ERROR: Problem retrieving data from Neo4j; query_rc = {query_rc}.\nExiting with return code 310.")
            # exit(310)
            return_msg = f"\nFATAL ERROR: Problem retrieving data from Neo4j; query_rc = {query_rc}.\nRC=310."
            return (310, return_msg, None)
        else:
            if _DEBUG_SWITCH:
                print(f"\nQuery result for: {each_inner_list}\n{query_result}\n")
            query_results_arr.append(query_result)
    
    ### COMMENTED FOR COMBINED PROGRAM LOGIC - start
    ## show results
    # print(f"\n\nFinal results=\n{query_results_arr}")
    # print(f"\n\nCOMPLETED NEO4J QUERY LOGIC.\n")
    ### COMMENTED FOR COMBINED PROGRAM LOGIC - end
    
    return (0, None, query_results_arr)

###############################################################################################################
## ----------------------  NEO4J QUERY LOGIC ENDS                 NEO4J QUERY LOGIC ENDS ---------------------- 
###############################################################################################################

## ----------------------------------------------------------------------------------------------------------------------

###############################################################################################################
## -----------  QUERY IMAGES SELECTION LOGIC STARTS             QUERY IMAGES SELECTION LOGIC STARTS ----------- 
###############################################################################################################

#global SAVED_KERAS_MODEL_PATH = r'/home/rohit/PyWDUbuntu/thesis/saved_keras_model/yolov3_coco80.saved.model'

class c_queryImgSelection_root_window:
    def __init__(self, _DEBUG_SWITCH, _query_num, _each_query_result, _index_positions_to_remove_this_query, _num_candidate_images_before_selection_began):
        self.DEBUG_SWITCH = _DEBUG_SWITCH
        self.query_num = _query_num
        self.each_query_result = _each_query_result
        ## this is the list passed from beginning. to be populated with positions of deselections during the
        ##      grid window processing
        self.index_positions_to_remove_this_query = _index_positions_to_remove_this_query
        
        self.msg_warning_for_root_do_not_close = r'DO NOT CLOSE THIS WINDOW TILL ALL SELECTIONS ARE COMPLETED FOR THIS QUERY!!!'
        self.msg_instructions_for_root = "".join([
            f"                               ----------------------------------",
            f"\n                               -----      INSTRUCTIONS      -----",
            f"\n                               ----------------------------------",
            f"\n",
            f"\n",
            f"\nThis is the main control window for candidate image selections for a particular query.",
            f"\n",
            f"\n\t1.  Click on the button to proceed.",
            f"\n",
            f"\n\t2.  A grid displays thumbnails of the candidate images from database query.",
            f"\n\t       Important: If there are no images to display you cannot click to proceed to the"
            f"\n\t                  grid selection window. Simply close this window and the next query"
            f"\n\t                  selections process will begin."
            f"\n",
            f"\n\t3.  By default, all the images are Selected.",
            f"\n\t       You can Deselect any images by clicking the image thumbnail.",
            f"\n\t       You can also Deselect all images.",
            f"\n\t       But you can only Select a maximum of 5 images.",
            f"\n\t       Monitor the count of currently Selected images.",
            f"\n",
            f"\n\t4.  Once ready with your Selections, click the button to Confirm Deselections.", 
            f"\n\t       NOTE: While the number of Selections is invalid (more than 5), you cannot",
            f"\n\t             click the button to confirm Selections.",
            f"\n",
            f"\n\t5.  You can Enlarge an image to inspect it more closely before deciding to Select or Deselect.",
            f"\n",
            f"\n\t6.  When viewing the Enlarged image, you can also perform object detection on the image to",
            f"\n\t         see an Inference image in a new window."
        ])

        ## root window for the query being processed
        self.root = tk.Tk()
        if _num_candidate_images_before_selection_began == 0:
            self.root.title(f"Image Selection - Query number {_query_num} - No Images to Display -- Please close this window")
        else:
            self.root.title(f"Image Selection - Query number {_query_num} - Click button to Proceed")
        ## label for warning message not to close
        self.lbl_root_msg_warning_not_close = tk.Label(
            master=self.root,
            text=self.msg_warning_for_root_do_not_close,
            bg="red", fg="white",
            width=(len(self.msg_warning_for_root_do_not_close) + 10),
            height=5
            )
        ## label for instructions
        self.lbl_root_instructions = tk.Label(
            master=self.root,
            text=self.msg_instructions_for_root,
            bg="blue", fg="white",
            justify=tk.LEFT,
            width=130,
            height=35,
            relief=tk.SUNKEN
            )
        ## button to proceed to grid selection window
        ## assume there are images and make it clickable
        self.btn_root_click_proceed = tk.Button(
            master=self.root,
            text=f"Click to view images and make Selections - Query {_query_num}",
            bg="yellow", fg="black",
            relief=tk.RAISED,
            borderwidth=15,
            command=partial(
                generic_show_grid_selection_window,
                self.root,
                self.each_query_result,
                self.query_num,
                self.index_positions_to_remove_this_query
                )
            )
        ## if no images to display in grid, disable the button to proceed and change the text displayed in button
        if _num_candidate_images_before_selection_began == 0:
            self.btn_root_click_proceed.configure(
            state=tk.DISABLED,
            relief=tk.FLAT,
            text=f"No images to make Selections - Query {_query_num} - Please close this window",
            )
        self.btn_root_click_proceed.configure(
            width=(len(self.btn_root_click_proceed["text"]) + 10),
            height=7
        )
        
        self.lbl_root_msg_warning_not_close.pack(padx=15, pady=15)
        self.lbl_root_instructions.pack(padx=10, pady=10)
        self.btn_root_click_proceed.pack(padx=50, pady=50)

def generic_show_grid_selection_window(_root, _each_query_result, _query_num, _index_positions_to_remove_this_query):
    
    ## dictionary with key as the image source, value is the location of that datasets images
    #source_and_location_image_datasets = {
    #    'flickr30k' : r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Data/flickr30k_images/flickr30k_images/',
    #    'coco_val_2017' : r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Data/COCO_val2017_5k/val2017/',
    #    'coco_test_2017' : r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/',
    #    'coco_train_2017' : r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Data/COCO_train2017_118k/'
    #    }
    source_and_location_image_datasets = {
        'flickr30k' : r'/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/flickr30k_images/flickr30k_images/',
        'coco_val_2017' : r'/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_val2017_5k/val2017/',
        'coco_test_2017' : r'/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/',
        'coco_train_2017' : r'/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_train2017_118k/'
        }
    
    ## build the list for images with their full path
    image_files_list = []
    for each_img_info in _each_query_result:
        image_files_list.append(
            os.path.join(
                source_and_location_image_datasets[each_img_info["Source"]],
                each_img_info["Image"]
                ))
    
    num_candidate_images_in_this_query = len(image_files_list)
    myStr = f"Num of images = {num_candidate_images_in_this_query}\narray=\n{image_files_list}\n"
    print_and_log(myStr, "debug")
    myStr = None

    opened_imgs_arr = []
    for idx in range(20):
        if idx < num_candidate_images_in_this_query:  ## image list has a valid entry to pick up
            img_orig = Image.open(image_files_list[idx])
            img_resized = img_orig.resize((100, 80),Image.ANTIALIAS)
            ## append tuple of resized image and file with path
            opened_imgs_arr.append( (ImageTk.PhotoImage(img_resized), image_files_list[idx]) )
            del img_orig, img_resized
        else:  ## there is no image
            opened_imgs_arr.append( (None, None) )
    
    ## the selection grid can handle maximum 20 image thumbnails.
    ##     There are 10 columns.
    ##     There are 2 rows of thumbnail images.
    ##     Each row of images has associated Enlarge buttons row below it.
    ##     So the selection window grid will have have 2 x 2 x 10 = 40 buttons.
    ##     In addition to these, there are two more tkinter widgets:
    ##        a Button for Confirm Selections, and
    ##        a Label to show live count of currently Selected images.
    ##     NOTE: We specify the number of rows and columns wrt the images. Logic will later assume
    ##           two additional rows for the Confirm button and Selection count Label.
    n_rows_images = 2
    n_cols_images = 10
    n_rows = 2 * 2 ## one for the images, one for the associated Enlarge button
    n_cols = n_cols_images

    ## make object for object detector - this same object will be used for inferennce on all images
    ##      in the grid. So no need to load new model for each inference.
    o_keras_inference_performer = c_keras_inference_performer()

    o_queryImgSelection_grid_wnd_window = c_queryImgSelection_grid_wnd_window(
        _root,
        _query_num,
        _index_positions_to_remove_this_query,
        num_candidate_images_in_this_query,
        n_rows,
        n_cols,
        opened_imgs_arr,  ## REMEMBER this is tuple of (Photo image object resized, image path)
        o_keras_inference_performer
        )
    
    o_queryImgSelection_grid_wnd_window.wnd_grid.mainloop()

class c_queryImgSelection_grid_wnd_window:
    def __init__(
        self,
        _root,
        _query_num,
        _index_positions_to_remove_this_query,
        _num_candidate_images_in_this_query,
        _n_rows,
        _n_cols,
        _opened_imgs_arr,
        _o_keras_inference_performer
        ):

        self.root = _root
        self.query_num = _query_num
        self.index_positions_to_remove_this_query = _index_positions_to_remove_this_query
        self.num_candidate_images_in_this_query_at_start = _num_candidate_images_in_this_query
        ## NOTE: This row count is only for thumbnail and Enlarge rows,
        ##       not for Confirm button and Label for current count.
        ##       They will be referenced by using row count and indexing further down automatically.
        self.n_rows = _n_rows   
        self.n_cols = _n_cols
        self.opened_imgs_arr = _opened_imgs_arr
        self.o_keras_inference_performer = _o_keras_inference_performer
        
        ## array for the buttons for thumbnail image button and the associated Enlarge button
        self.grid_buttons_arr = []
        ## The number of Selected candidate images cannot be more than this limit
        self.max_limit_images_selections = 5
        ## initialise the current count of selections to the number of images at start
        self.images_selected_current_count = self.num_candidate_images_in_this_query_at_start

        ## window for the grid selection
        self.wnd_grid = tk.Toplevel(master=_root)
        self.wnd_grid.title(f"Thumbnails and Selection Window -- Query number {self.query_num}")
        ## label for count of current selections
        self.lbl_track_selected_count = tk.Label(
            master=self.wnd_grid,
            text="",
            relief=tk.FLAT,
            borderwidth=10
            )
        ## call function to update the text with count, and the color 
        self.update_label_count_currently_selected_images()
        ## button for selection confirm
        self.btn_confirm_selections = tk.Button(
            master=self.wnd_grid,
            text=f"Click to Confirm Deselections -- clickable only if current selection count is <= {self.max_limit_images_selections}",
            bg="yellow", fg="black",
            borderwidth=10,
            relief=tk.RAISED,
            command=self.do_confirm_selections_processing
            )
        self.btn_confirm_selections.configure(
            width=( len(self.btn_confirm_selections["text"]) + 20 ),
            height=5
            )
        ## prevent Confirm Selections if there are too many images Selected at start
        if self.images_selected_current_count > self.max_limit_images_selections:
            self.btn_confirm_selections.configure(state=tk.DISABLED, relief=tk.FLAT)
        
        ## populate the button array for the grid - thumbnail and Enlarge buttons
        ##    first make skeleton entries for the buttons
        ##    by default assume no Image is present to display,
        ##       so all buttons are in disabled state with the text saying "No Image"
        for r_idx in range(self.n_rows):
            self.wnd_grid.columnconfigure(r_idx, weight=1, minsize=50)
            self.wnd_grid.rowconfigure(r_idx, weight=1, minsize=50)
            temp_row_data = []
            for c_idx in range(self.n_cols):
                ## alternate row entries for button of image thumbnail and Enlarge
                if r_idx % 2 == 0:
                    ## thumbnail image button
                    temp_row_data.append(
                        tk.Button(
                            master=self.wnd_grid,
                            text="No Image",
                            bg="black", fg="white",
                            relief=tk.FLAT,
                            borderwidth=10,
                            state=tk.DISABLED
                            )
                        )
                else:
                    ## Enlarge button type
                    temp_row_data.append(
                        tk.Button(
                            master=self.wnd_grid,
                            text="Enlarge",
                            bg="black", fg="white",
                            relief=tk.FLAT,
                            borderwidth=10,
                            state=tk.DISABLED
                            )
                        )
            self.grid_buttons_arr.append(temp_row_data)

        ## now populate the Images and activate both button where applicable
        img_idx = 0 ## index to access each image from the tuple of (image, path)
        for r_idx in range(self.n_rows):
            for c_idx in range(self.n_cols):
                ## set grid position for all the label elements
                self.grid_buttons_arr[r_idx][c_idx].grid(
                    row=r_idx, column=c_idx,
                    padx=5, pady=5,
                    sticky="nsew"
                )
                ## only for the thumbnail rows, populate the images if it is available
                ##      if yes, change the state of thumbnail and associated Enlarge buttons
                if (r_idx % 2 == 0) and (img_idx < self.num_candidate_images_in_this_query_at_start):
                    ## r_idx is for an image thumbnail row and there is an image to show
                    ## from the input tuple extract the image and the path
                    resized_image, self.grid_buttons_arr[r_idx + 1][c_idx].image_path = self.opened_imgs_arr[img_idx]
                    self.grid_buttons_arr[r_idx][c_idx].image = None
                    self.grid_buttons_arr[r_idx][c_idx].configure(
                            image=resized_image,
                            relief=tk.SUNKEN,
                            borderwidth=10,
                            highlightthickness = 15,
                            highlightbackground = "green", highlightcolor= "green",
                            state=tk.NORMAL,
                            command=partial(
                                self.do_image_select_button_clicked_processing,
                                r_idx, c_idx
                                )
                        )
                    img_idx += 1
                    ## make variable to hold an Enlarged image window object for the associated Enlarge button.
                    ##     set as None for now.
                    ##     if associated Enlarge button is clicked, object will be populated and used.
                    self.grid_buttons_arr[r_idx + 1][c_idx].o_EnlargeImage_window = None
                    ## change the associated Enlarge button
                    self.grid_buttons_arr[r_idx + 1][c_idx].configure(
                            relief=tk.RAISED,
                            borderwidth=10,
                            state=tk.NORMAL,
                            command=partial(
                                generic_show_enlarged_image_window,
                                self.wnd_grid,
                                self.grid_buttons_arr[r_idx + 1][c_idx].o_EnlargeImage_window,
                                self.grid_buttons_arr[r_idx + 1][c_idx].image_path,
                                self.o_keras_inference_performer
                                )
                        )
        
        ## label for count of current selections
        r_idx = self.n_rows
        c_idx = 0
        self.lbl_track_selected_count.grid(
            row=r_idx, column=c_idx,
            rowspan=1, columnspan=self.n_cols,
            sticky="nsew",
            padx=5, pady=5
            )

        ## button for selection confirm
        r_idx = self.n_rows + 1
        c_idx = 0
        self.btn_confirm_selections.grid(
            row=r_idx, column=c_idx,
            rowspan=1, columnspan=self.n_cols,
            sticky="nsew",
            padx=5, pady=5
            )
        
        return
    
    def do_confirm_selections_processing(self):
        ## For the images that are Deselected, figure out the position and add the position number
        ##     to the return list.
        self.index_positions_to_remove_this_query
        for r_idx in range(0, self.n_rows, 2):
            for c_idx in range(self.n_cols):
                if self.grid_buttons_arr[r_idx][c_idx]["relief"] == tk.RAISED:
                    ## this image is Deselected - so extract the position
                    self.index_positions_to_remove_this_query.append( ( (r_idx // 2) * self.n_cols ) + c_idx )
                
        myStr = f"\nFor Query {self.query_num}, Deselected positions=\n{self.index_positions_to_remove_this_query}"
        print_and_log(myStr, "info")
        myStr = None
        
        self.root.destroy()
        return

    def update_label_count_currently_selected_images(self):
        ## update the count based on latest count of selected images
        ##        also change the color if the count is greater than allowed limit
        self.lbl_track_selected_count.configure(
            text=" ".join([ "Count of Images currently Selected =", str(self.images_selected_current_count) ])
            )
        self.lbl_track_selected_count.configure(
            width=( len(self.lbl_track_selected_count["text"]) + 10 ),
            height=5
            )
        if self.images_selected_current_count > self.max_limit_images_selections:
            self.lbl_track_selected_count.configure(bg="red", fg="white")
        else:
            self.lbl_track_selected_count.configure(bg="green", fg="white")
        return

    def do_image_select_button_clicked_processing(self, _r_idx, _c_idx):
        ## toggle button characteristics:
        ##                              Relief         Color around image
        ##       Selected   Image       SUNKEN         Green
        ##       Deselected Image       RAISED         Red
        if self.grid_buttons_arr[_r_idx][_c_idx]["relief"] == tk.SUNKEN:
            ## Image is currently Selected, change to Deselected
            self.grid_buttons_arr[_r_idx][_c_idx].configure(
                relief=tk.RAISED,
                highlightbackground = "red", highlightcolor= "red"
            )
            self.images_selected_current_count -= 1
        else:
            ## Image is currently Deselected, change to Selected
            self.grid_buttons_arr[_r_idx][_c_idx].configure(
                relief=tk.SUNKEN,
                highlightbackground = "green", highlightcolor= "green"
            )
            self.images_selected_current_count += 1
        ## update the label for count
        self.update_label_count_currently_selected_images()
        ## update the confirm button characteristics
        if self.images_selected_current_count > self.max_limit_images_selections:
            self.btn_confirm_selections.configure(state=tk.DISABLED, relief=tk.FLAT)
        else:
            self.btn_confirm_selections.configure(state=tk.NORMAL, relief=tk.RAISED)
        return

def generic_show_enlarged_image_window(_wnd_grid, _o_EnlargeImage_window, _image_path, _o_keras_inference_performer):
    ## make the object for Enlarged window and show the window. Currently object is None
    _o_EnlargeImage_window = c_EnlargeImage_window( _wnd_grid, _image_path, _o_keras_inference_performer)
    _o_EnlargeImage_window.wnd_enlarged_img.mainloop()

class c_EnlargeImage_window:
    def __init__(self, _master_wnd, _image_path, _o_keras_inference_performer):
        self.master = _master_wnd
        self.image_path = _image_path
        self.o_keras_inference_performer = _o_keras_inference_performer

        self.img_orig = ImageTk.PhotoImage(Image.open(self.image_path))

        ## window for the Enlarged image
        self.wnd_enlarged_img = tk.Toplevel(master=self.master)
        self.wnd_enlarged_img.title(f"Enlarged image: {os.path.basename(self.image_path)}")
        ## label for Enlarged image
        self.lbl_enlarged_img = tk.Label(
            master=self.wnd_enlarged_img,
            relief=tk.FLAT,
            borderwidth=10,
            image=self.img_orig)
        self.lbl_enlarged_img_full_path = tk.Label(
            master=self.wnd_enlarged_img,
            relief=tk.SUNKEN,
            borderwidth=5,
            bg="grey", fg="white",
            text=f"Image path = {self.image_path}"
            )
        self.lbl_enlarged_img_full_path.configure(
            width=( len(self.lbl_enlarged_img_full_path["text"]) + 10),
            height=3
            )
        ## button for Inference
        self.btn_enlarged_img_do_inference = tk.Button(
            master=self.wnd_enlarged_img,
            relief=tk.RAISED,
            borderwidth=10,
            text="Click to perform object detection inference on this image",
            bg="yellow", fg="black",
            command=self.do_inference_and_display_output
            )
        self.btn_enlarged_img_do_inference.configure(
            width=( len(self.btn_enlarged_img_do_inference["text"]) + 10 ),
            height=3
            )
        
        self.lbl_enlarged_img.pack(padx=15, pady=15)
        self.lbl_enlarged_img_full_path.pack(padx=15, pady=15)
        self.btn_enlarged_img_do_inference.pack(padx=15, pady=15)
    
    def do_inference_and_display_output(self):
        myStr = f"\n\nInference invoked for: {self.image_path}\n"
        print_and_log(myStr, "debug")
        myStr = None

        ## window for inference
        self.wnd_inference_img = tk.Toplevel(master=self.wnd_enlarged_img)
        self.wnd_inference_img.title(f"Inference for: {os.path.basename(self.image_path)}")
        ## label for the output image after inference - set as None for now
        self.lbl_inference_img = tk.Label(
            master=self.wnd_inference_img,
            relief=tk.FLAT,
            borderwidth=4,
            image=None
            )
        ## label for the image path
        self.lbl_path_img_inferred = tk.Label(
            master=self.wnd_inference_img,
            relief=tk.SUNKEN,
            borderwidth=5,
            bg="grey", fg="white",
            text=f"Image path = {self.image_path}"
            )
        self.lbl_path_img_inferred.configure(
            width=( len(self.lbl_path_img_inferred["text"]) + 10),
            height=3
            )
        ## label for the inference info of types of objects found
        self.lbl_inference_textual_info = tk.Label(
            master=self.wnd_inference_img,
            relief=tk.SUNKEN,
            borderwidth=5,
            bg="blue", fg="white",
            justify='left'
            )
        ## do the actual inference by saved keras model, then superimpose the bounding boxes and write the
        ##    output image to an intermediate file.
        ## But first, make empty list to to be populated by model inference performer.
        ##       entries made will be f-string of name of objects found with percentage
        ##            without the new-line character at end.
        self.objects_found_info_arr = []
        self.o_keras_inference_performer.perform_model_inference(self.image_path, self.objects_found_info_arr)
        ## reload the intermediate file for tkinter to display, then delete the file
        inference_model_output_image_path = r'./intermediate_file_inferenece_image.jpg'
        inference_model_output_img = ImageTk.PhotoImage(Image.open(inference_model_output_image_path))
        os.remove(inference_model_output_image_path)

        ## put the inference outpu image in the label widget
        self.lbl_inference_img.configure(image=inference_model_output_img)
        ## extract the info about objects found and put into the label text
        textual_info_to_display = "\n".join( self.objects_found_info_arr )
        self.lbl_inference_textual_info.configure(text=textual_info_to_display)
        self.lbl_inference_textual_info.configure(
            width=( 20 + max( [len(line_text) for line_text in self.objects_found_info_arr] )  ),
            height=(2 + len(self.objects_found_info_arr) )
            )
        
        ## pack it all
        self.lbl_inference_img.pack(padx=15, pady=15)
        self.lbl_inference_textual_info.pack(padx=15, pady=15)
        self.lbl_path_img_inferred.pack(padx=15, pady=15)

        ## display the Inference window
        self.wnd_inference_img.mainloop()
        return

class c_keras_inference_performer: ## adapted from jbrownlee/keras-yolo3
    
    def __init__(self):
        self.reloaded_yolo_model = None
    
    class BoundBox:
        def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
            self.xmin = xmin
            self.ymin = ymin
            self.xmax = xmax
            self.ymax = ymax
            
            self.objness = objness
            self.classes = classes

            self.label = -1
            self.score = -1
        
        def get_label(self):
            if self.label == -1:
                self.label = np.argmax(self.classes)
            return self.label
        
        def get_score(self):
            if self.score == -1:
                self.score = self.classes[self.get_label()]  
            return self.score
    
    def _interval_overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b

        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2,x4) - x3
    
    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def bbox_iou(self, box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
        
        intersect = intersect_w * intersect_h

        w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
        w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
        
        union = w1*h1 + w2*h2 - intersect
        
        return float(intersect) / union
    
    def preprocess_input(self, image, net_h, net_w):
        new_h, new_w, _ = image.shape

        # determine the new size of the image
        if (float(net_w)/new_w) < (float(net_h)/new_h):
            new_h = (new_h * net_w)/new_w
            new_w = net_w
        else:
            new_w = (new_w * net_h)/new_h
            new_h = net_h
        
        # resize the image to the new size
        resized = cv2.resize(image[:,:,::-1]/255., (int(new_w), int(new_h)))

        # embed the image into the standard letter box
        new_image = np.ones((net_h, net_w, 3)) * 0.5
        new_image[int((net_h-new_h)//2):int((net_h+new_h)//2), int((net_w-new_w)//2):int((net_w+new_w)//2), :] = resized
        new_image = np.expand_dims(new_image, 0)

        return new_image
    
    def decode_netout(self, netout, anchors, obj_thresh, nms_thresh, net_h, net_w):
        grid_h, grid_w = netout.shape[:2]
        nb_box = 3
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))
        nb_class = netout.shape[-1] - 5

        boxes = []

        netout[..., :2]  = self._sigmoid(netout[..., :2])
        netout[..., 4:]  = self._sigmoid(netout[..., 4:])
        netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
        netout[..., 5:] *= netout[..., 5:] > obj_thresh

        for i in range(grid_h*grid_w):
            row = i / grid_w
            col = i % grid_w
            
            for b in range(nb_box):
                # 4th element is objectness score
                objectness = netout[int(row)][int(col)][b][4]
                #objectness = netout[..., :4]
                
                if(objectness.all() <= obj_thresh): continue
                
                # first 4 elements are x, y, w, and h
                x, y, w, h = netout[int(row)][int(col)][b][:4]

                x = (col + x) / grid_w # center position, unit: image width
                y = (row + y) / grid_h # center position, unit: image height
                w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
                h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height  
                
                # last elements are class probabilities
                classes = netout[int(row)][col][b][5:]
                
                box = self.BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)

                boxes.append(box)
        
        return boxes
    
    def correct_yolo_boxes(self, boxes, image_h, image_w, net_h, net_w):
        if (float(net_w)/image_w) < (float(net_h)/image_h):
            new_w = net_w
            new_h = (image_h*net_w)/image_w
        else:
            new_h = net_w
            new_w = (image_w*net_h)/image_h
        
        for i in range(len(boxes)):
            x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
            y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
            
            boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
            boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
            boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
            boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
    
    def do_nms(self, boxes, nms_thresh):
        if len(boxes) > 0:
            nb_class = len(boxes[0].classes)
        else:
            return
        
        for c in range(nb_class):
            sorted_indices = np.argsort([-box.classes[c] for box in boxes])

            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]

                if boxes[index_i].classes[c] == 0: continue

                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if self.bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                        boxes[index_j].classes[c] = 0
    
    def draw_boxes(self, image, boxes, labels, obj_thresh, _objects_found_info_arr):
        serial_num_object_box_overall = 1
        for box in boxes:
            label_str = ''
            label = -1
            serial_num_object_box_at_loop_start = serial_num_object_box_overall
            for i in range(len(labels)):
                if box.classes[i] > obj_thresh:
                    label_str += labels[i]
                    label = i

                    #print(labels[i] + ': ' + str(box.classes[i]*100) + '%') - original code

                    myStr = f"{serial_num_object_box_overall}) {labels[i]} : {box.classes[i]*100:.2f}%"
                    print_and_log(myStr, "info")
                    myStr = None
                    _objects_found_info_arr.append( f"{serial_num_object_box_overall}) {labels[i]} :\t\t{box.classes[i]*100:.2f} %" )
                    serial_num_object_box_overall += 1
            
            if label >= 0:
                cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), (0,255,0), 3)
                # cv2.putText(image, 
                #             label_str + ' ' + str(box.get_score()), 
                #             (box.xmin, box.ymin - 13), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 
                #             1e-3 * image.shape[0], 
                #             (0,255,0), 2)
                text_to_superimpose_in_image = f"{serial_num_object_box_at_loop_start}) {label_str}"
                cv2.putText(image, 
                            text_to_superimpose_in_image, 
                            (box.xmin, box.ymin - 13), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1e-3 * image.shape[0], 
                            (0,255,0), 2)
                serial_num_object_box_at_loop_start += 1
        return image
    
    def perform_model_inference(self, _path_infer_this_image, _objects_found_info_arr):
        myStr = f"\n\nExecuting model inference on image_to_infer : {_path_infer_this_image}\n"
        print_and_log(myStr, "debug")
        myStr = None
        
        ## load the keras pretained model if not already loaded
        if self.reloaded_yolo_model is None:
            saved_model_location = r'/home/rohit/PyWDUbuntu/thesis/saved_keras_model/yolov3_coco80.saved.model'
            self.reloaded_yolo_model = keras_load_model(saved_model_location)
            myStr = f"\n\n    LOADED KERAS MODEL from: {saved_model_location}      \n\n"
            print_and_log(myStr, "info")
            myStr = None
        
        ## set some parameters for network
        net_h, net_w = 416, 416
        obj_thresh, nms_thresh = 0.5, 0.45
        anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
        ## list of object classes the object detector model is trained on
        labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
                "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
                "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
                "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
                "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
                "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        
        image_to_infer_cv2 = cv2.imread(_path_infer_this_image)
        image_h, image_w, _ = image_to_infer_cv2.shape
        try:
            image_to_infer_preprocessed = self.preprocess_input(image_to_infer_cv2, net_h, net_w)
        except Exception as error_inference_preprocess_image:
            #print(f"\nFATAL ERROR: Problem reading the input file.\nError message: {error_inference_preprocess_image}\nExit RC=400")
            myStr = f"\nFATAL ERROR: Problem reading the input file.\nError message: {error_inference_preprocess_image}\nExit RC=400"
            print_and_log(myStr, "error")
            myStr = None
            exit(400)
        
        ## run the prediction
        yolos = self.reloaded_yolo_model.predict(image_to_infer_preprocessed)
        boxes = []

        for i in range(len(yolos)):
            ## decode the output of the network
            boxes += self.decode_netout(yolos[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)
        
        ## correct the sizes of the bounding boxes
        self.correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

        ## suppress non-maximal boxes
        self.do_nms(boxes, nms_thresh)

        ## draw bounding boxes into the image
        self.draw_boxes(image_to_infer_cv2, boxes, labels, obj_thresh, _objects_found_info_arr)

        ## save the image as intermediate file -- see later whether to return and processing is possible
        cv2.imwrite(r'./intermediate_file_inferenece_image.jpg', (image_to_infer_cv2).astype('uint8') )

def select_images_functionality_for_one_query_result(_DEBUG_SWITCH, _query_num, _each_query_result, _index_positions_to_remove_this_query, _num_candidate_images_before_selection_began):
    
    ## make a root window and show it
    o_queryImgSelection_root_window = c_queryImgSelection_root_window(_DEBUG_SWITCH, _query_num, _each_query_result, _index_positions_to_remove_this_query, _num_candidate_images_before_selection_began)
    o_queryImgSelection_root_window.root.mainloop()

def gui_candidate_image_selection_functionality(_query_neo4j_module_results, _DEBUG_SWITCH):
    '''
    High level module to present images returned by neo4j query via a graphical user interface.
        Allow Deselection of images for two purposes:
            1) images are wrongly picked up (as object of interest is actually not present)
            2) Neo4j query can returnup to 20 images per query. But the maximum limit of images per query
               to pass to the next stage (auto-caption block) is only 5.
        Allow viewing Enlarged image to study clearly. Also, allow object detection inference again.
               Thus provide more information to user to make Deselection decision.
        Show live count of number of images currently Selected (default start is with all images Selected).
        Allow clicking of Deslections Confirmation only if the count of Selected images is within the max.
               permissible limit.
    VARIABLES PASSED:
          1) Data structure containing results of queries to Neo4j database - consists of image name and the
             source name to indicate which dataset the image belongs to.
          2) Debugging switch
    RETURNS:
          1) Return code
                value of 0 => all ok
                non 0      => some problem
          2) Message if return code is non-zero, else will be None
          3) Results array - same data struture as the input but containing only the Selected images.
                Per query, maximum 5 images information will be present.
                NOTE: It could be 0 too, if user decided to Deselect all the images.
    ''' 
    # ## test data for unit test - actually will be passed from earlier stage (id keyword elements)
    # database_query_results = [
    #     [
    #         {'Image': '000000169542.jpg', 'Source': 'coco_test_2017'},
    #         {'Image': '000000169516.jpg', 'Source': 'coco_test_2017'},
    #         {'Image': '000000313777.jpg', 'Source': 'coco_test_2017'},
    #         {'Image': '000000449668.jpg', 'Source': 'coco_test_2017'},
    #         {'Image': '000000292186.jpg', 'Source': 'coco_test_2017'},
    #         {'Image': '000000168815.jpg', 'Source': 'coco_test_2017'},
    #         {'Image': '000000168743.jpg', 'Source': 'coco_test_2017'}
    #         ],
    #     [
    #         {'Image': '000000146747.jpg', 'Source': 'coco_test_2017'},
    #         {'Image': '000000509771.jpg', 'Source': 'coco_test_2017'}
    #         ],
    #     [
    #         {'Image': '000000012149.jpg', 'Source': 'coco_test_2017'}
    #         ]
    #     ]
    
    ## execute gui logic to allow deselection of images, check the number of selections are valid,
    ##         capture the deselected positions
    index_positions_to_remove_all_queries = []
    for query_num, each_query_result in enumerate(_query_neo4j_module_results):
        num_candidate_images_before_selection_began = len(each_query_result)
        index_positions_to_remove_this_query = []

        select_images_functionality_for_one_query_result(_DEBUG_SWITCH, query_num + 1, each_query_result, index_positions_to_remove_this_query, num_candidate_images_before_selection_began)
        index_positions_to_remove_all_queries.append(index_positions_to_remove_this_query)
        
        num_images_to_remove = len(index_positions_to_remove_this_query)
        myStr = "\n".join([
            f"\nCompleted selection process - Query number {query_num + 1}",
            f"Number of images before selection began = {num_candidate_images_before_selection_began}",
            f"Number of images Deselected by user = {num_images_to_remove}.",
            f"Number of images that will remain = { num_candidate_images_before_selection_began - num_images_to_remove }\n"
            ])
        print_and_log(myStr, "debug")
        myStr = None
        
    ## remove the Deselected images
    final_remaining_images_selected_info = []
    for each_query_result, each_index_positions_to_remove in \
        zip(_query_neo4j_module_results, index_positions_to_remove_all_queries):
        temp_array = [each_image_info for idx, each_image_info in enumerate(each_query_result) if idx not in each_index_positions_to_remove]
        final_remaining_images_selected_info.append(temp_array)
    
    ## show summary info
    myStr = f"\n\n-------------------------------- SUMMARY INFORMATON --------------------------------\n"
    print_and_log(myStr, "debug")
    myStr = None

    for query_num, (each_query_results, each_query_final_remaining_images_info, each_query_index_positions_remove) in \
        enumerate(zip(_query_neo4j_module_results, final_remaining_images_selected_info, index_positions_to_remove_all_queries)):
        myStr = "\n".join([
            f"\nFOR QUERY {query_num + 1}\nNumber of candidate images before selection = {len(each_query_results)}",
            f"Number of Deselections done = {len(each_query_index_positions_remove)}",
            f"Number of images remaining after Deselections = {len(each_query_final_remaining_images_info)}\n"
            ])
        print_and_log(myStr, "debug")
        myStr = None

        myStr = "\n".join([
            f"\n\t------ Query images info BEFORE::\n{each_query_results}",
            f"\t------ Positions removed::\n{each_query_index_positions_remove}",
            f"\t------ Query images info AFTER::\n{each_query_final_remaining_images_info}\n\n"
            ])
        print_and_log(myStr, "debug")
        myStr = None
    
    return (0, None, final_remaining_images_selected_info)

###############################################################################################################
## -----------    QUERY IMAGES SELECTION LOGIC ENDS               QUERY IMAGES SELECTION LOGIC ENDS ----------- 
###############################################################################################################

## ----------------------------------------------------------------------------------------------------------------------

###############################################################################################################
## -----   IMAGE CAPTONING - NO ATTN - INFERENCE STARTS      IMAGE CAPTONING - NO ATTN - INFERENCE STARTS -----
###############################################################################################################

def preprocess_image_for_Incepv3(_img_path, _key = 'DUMMY', _DEBUG_SWITCH=False):
    """
    ## Make images suitable for use by Inception-v3 model later
    ##
    ## Resize to (299, 299)
    ## As model needs 4-dim input tensor, add one dimenion to make it (1, 299, 299, 3)
    ## Preprocess the image using custom function of Inception-v3 model
    """
    img = tf.keras.preprocessing.image.load_img(_img_path, target_size=(299, 299))
    #print(f"type={type(img)}") # type(img): type=<class 'PIL.Image.Image'>
    if _DEBUG_SWITCH:
        plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.title('Original Image(Resized): ' + _key + '.jpg')
        plt.imshow(img)
    img = tf.keras.preprocessing.image.img_to_array(img) # Converts PIL Image instance to numpy array (299,299,3)
    img = np.expand_dims(img, axis=0) #Add one more dimension: (1, 299, 299, 3) # Inception-V3 requires 4 dimensions
    img = tf.keras.applications.inception_v3.preprocess_input(img) # preprocess image as per Inception-V3 model
    if _DEBUG_SWITCH:
        plt.subplot(122)
        plt.title('Preprocessed image for Inception-V3: ' + _key + '.jpg')
        plt.imshow(img[0])
    return img  # shape will be (1, 299, 299, 3)

def encode_image(_imgpath, _model_CNN_encoder, _key = 'DUMMY', _DEBUG_SWITCH = False):
    """
    # Function to encode given image into a vector of size (2048, )
    """
    preproc_img = preprocess_image_for_Incepv3(_imgpath, _key = 'DUMMY', _DEBUG_SWITCH = False) # preprocess image per Inception-v3 requirements
    encoded_features = _model_CNN_encoder.predict(preproc_img) # Get encoding vector for image
    encoded_features = encoded_features.reshape(encoded_features.shape[1], ) # reshape from (1, 2048) to (2048, )
    return encoded_features

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
        
        if _DEBUG_SWITCH:
            print(f"\nRNN Decoder model defined with these paramenters:\nEMBEDDING_DIMS = {_EMBEDDING_DIMS} , VOCAB_SIZE = {_VOCAB_SIZE} , MAX_LENGTH_CAPTION = {_MAX_LENGTH_CAPTION}\nAttempting to load weights...")
        myStr = "\n".join([
            f"\nRNN Decoder model defined with these paramenters:",
            f"\nEMBEDDING_DIMS = {_EMBEDDING_DIMS} , VOCAB_SIZE = {_VOCAB_SIZE} , MAX_LENGTH_CAPTION = {_MAX_LENGTH_CAPTION}",
            f"\nAttempting to load weights..."
            ])
        print_and_log(myStr, "info")
        myStr = None
        
        ## load the weights
        reloaded_rnn_decoder_model.load_weights(_saved_weights_file)

        if _DEBUG_SWITCH:
            print(f"SUCCESS - Reloaded weights from :: {_saved_weights_file}")

        return reloaded_rnn_decoder_model
    else:
        #print(f"\nERROR reloading weights. Check weights file exists here = {_saved_weights_file} ;\nOR model setup parameters incompatible with the saved weights file given.")
        myStr = "\n".join([
            f"\nERROR reloading weights. Check weights file exists here = {_saved_weights_file} ;",
            f"\nOR model setup parameters incompatible with the saved weights file given."
            ])
        print_and_log(myStr, "error")
        myStr = None

        return None

def greedySearch(_decoder_model, _img_encoding, _max_length, _wordtoix = None, _ixtoword = None):
    wordtoix = _wordtoix
    ixtoword = _ixtoword
    in_text = 'startseq'
    for i in range(_max_length):
        sequence = [ wordtoix[w] for w in in_text.split() if w in wordtoix ]
        sequence = keras.preprocessing.sequence.pad_sequences([sequence], maxlen=_max_length)
        yhat = _decoder_model.predict([_img_encoding,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    caption_out = in_text.split()
    #caption_out = caption_out[1:-1]  ## drop the startseq and endseq words at either end
    caption_out = ' '.join(caption_out)
    return caption_out

## per link: https://yashk2810.github.io/Image-Captioning-using-InceptionV3-and-Beam-Search/
##     "Image Captioning using InceptionV3 and Beam Search"
def beamSearch(_decoder_model, _img_encoding, _max_length, _wordtoix = None, _ixtoword = None, _beam_width=3):
    start = [_wordtoix["startseq"]]
    
    # start_word[0][0] = index of the starting word
    # start_word[0][1] = probability of the word predicted
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < _max_length:
        temp = []
        for s in start_word:
            sequence = keras.preprocessing.sequence.pad_sequences([s[0]], maxlen=_max_length, padding='post')
            preds = _decoder_model.predict([_img_encoding,sequence], verbose=0)
            #print(f"\n\ntype(preds) = {type(preds)}\npreds=\n{preds}\n\n")
            
            # Getting the top <_beam_width>(n) predictions
            word_preds = np.argsort(preds[0])[-_beam_width:]
            
            # creating a new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-_beam_width:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [_ixtoword[i] for i in start_word]

    final_caption = []

    for i in intermediate_caption:
        final_caption.append(i)
        if i == "endseq":
            break
    
    final_caption = ' '.join(final_caption) ## retain the startseq and endseq token words
    return final_caption

def do_inference_one_image_totally_new_with_BeamSearch(_infer_image_jpg, _img_encoding, _MAX_LENGTH_CAPTION, _model_RNN_decoder, _imgcap_inference_beam_width, _DEBUG_SWITCH = False, _wordtoix = None, _ixtoword = None ):
    ## show the original image
    image = _img_encoding.reshape((1,2048))
    
    ## briefly show the image about to be inferred if in debug mode
    if _DEBUG_SWITCH:
        plt.ion()
        x = plt.imread(_infer_image_jpg)
        plt.imshow(x)
        plt.show()
        plt.pause(1.5)  ## display for 1.5 seconds
    
    predicted_caption = None
    ## get the prediction caption using greedy search
    predicted_caption = beamSearch(_model_RNN_decoder, image, _MAX_LENGTH_CAPTION, _wordtoix = _wordtoix, _ixtoword = _ixtoword, _beam_width=_imgcap_inference_beam_width)
    
    if _DEBUG_SWITCH:
        print(f"\nFor image :: {_infer_image_jpg}\n\nInference caption output:\n{ predicted_caption }")
    return predicted_caption

def do_inference_one_image_totally_new_with_GreedySearch(_infer_image_jpg, _img_encoding, _MAX_LENGTH_CAPTION, _model_RNN_decoder, _DEBUG_SWITCH = False, _wordtoix = None, _ixtoword = None ):
    ## show the original image
    image = _img_encoding.reshape((1,2048))
    
    ## briefly show the image about to be inferred if in debug mode
    if _DEBUG_SWITCH:
        plt.ion()
        x = plt.imread(_infer_image_jpg)
        plt.imshow(x)
        plt.show()
        plt.pause(1.5)  ## display for 1.5 seconds
    
    predicted_caption = None
    ## get the prediction caption using greedy search
    predicted_caption = greedySearch(_model_RNN_decoder, image, _MAX_LENGTH_CAPTION, _wordtoix = _wordtoix, _ixtoword = _ixtoword)
    
    if _DEBUG_SWITCH:
        print(f"\nFor image :: {_infer_image_jpg}\n\nInference caption output:\n{ predicted_caption }")
    return predicted_caption

def infer_for_caption_one_image(_in_img_jpg, _MAX_LENGTH_CAPTION, _model_CNN_encoder, _model_RNN_decoder, _imgcap_inference_beam_width, _wordtoix = None, _ixtoword = None, _DEBUG_SWITCH = False):
    ## Get the encoding by running thru bottleneck
    ## Function to encode given image into a vector of size (2048, )
    img_encoding_for_inference = encode_image(_in_img_jpg, _model_CNN_encoder, _DEBUG_SWITCH)
    
    if _DEBUG_SWITCH:
        print(f"Encoding shape = {img_encoding_for_inference.shape}")
    
    ## now do the decoder inference using the encoding - changed from GreedySearch to BeamSearch
    #predicted_caption = do_inference_one_image_totally_new_with_GreedySearch(_in_img_jpg, img_encoding_for_inference, _MAX_LENGTH_CAPTION, _model_RNN_decoder, _DEBUG_SWITCH, _wordtoix = _wordtoix, _ixtoword = _ixtoword )
    predicted_caption = do_inference_one_image_totally_new_with_BeamSearch(_in_img_jpg, img_encoding_for_inference, _MAX_LENGTH_CAPTION, _model_RNN_decoder, _imgcap_inference_beam_width, _DEBUG_SWITCH, _wordtoix = _wordtoix, _ixtoword = _ixtoword )
    return predicted_caption

def convert_prev_stages_data(_id_elements_module_results, _gui_candidate_image_selection_module_results):
    """
    ## Explanation of the data preparation using previous stage outputs:
    ## The expected output after processing the earlier stage outputs is a list of exactly 3 items.
    ##     One item corresponding to each of the original input sentences.
    ##     Each of these items in the list is a dict with two keys: 'key_elements' and 'selected_images'

    ## 'key_elements' is a list of objects selected by user at end of the Id Key Elements stage
    ##      - 0 to 3 values - can be empty list too
    ## 'selected_images' is a list containing lists. The inner list represents each of the images finally 
    ##      selected by the user at end GUI Selection of Images that the Neo4j query had returned.
    ##      - 0 to 5 images - can be empty list too
    ##      Regarding innner list: first entry is the image path, second is initialised as None to be filled 
    ##          later with the caption after processing the image through this stage.
    ##          Thus, ['/full/path/to/image.jpg' , None] will become something like
    ##                ['/full/path/to/image.jpg' , 'caption of the images after processing']

    ## Example of how the data structure could be in this scenario:
    ## For input sentence 1, the user finally selected 2 Key Elements, then out of the up to 20 images
    ##     returned by the neo4j query, user selected only 3 images to send to captioning stage while
    ##     up to 5 could have been selected.
    ## For input sentence 2, the user finally selected 0 Key Elements, thus there were no images
    ##     returned by the neo4j query, and nothing for the user to select and to send to captioning stage.
    ## For input sentence 3, the user finally selected 3 Key Elements, then out of the up to 20 images
    ##     returned by the neo4j query, user selected 0 images to send to captioning stage while
    ##     up to 5 could have been selected.
    ## -----------------------------------------------------------------------------
    example_data_structure_to_be_prepared_by_this_function = \
    [
        {
            'key_elements' : ['q1_obj1', 'q1_obj2'],
            'selected_images_no_attn' : [
                ['/path/to/q1_image1.jpg' , None],
                ['/path/to/q1_image2.jpg' , None],
                ['/path/to/q1_image3.jpg' , None]
            ]
        },
        {
            'key_elements' : [],
            'selected_images_no_attn' : []
        },
        {
            'key_elements' : ['q3_obj1', 'q3_obj2', 'q3_obj3'],
            'selected_images_no_attn' : []
        }
    ]
    ## -----------------------------------------------------------------------------
    """

    ## dictionary with key as the image source, value is the location of that datasets images
    source_and_location_image_datasets = {
        'flickr30k' : r'/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/flickr30k_images/flickr30k_images/',
        'coco_val_2017' : r'/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_val2017_5k/val2017/',
        'coco_test_2017' : r'/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/',
        'coco_train_2017' : r'/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_train2017_118k/'
        }
    
    ## add the full path to the image files in a NEW data structure copy
    new_gui_candidate_image_selection_module_results = copy.deepcopy(_gui_candidate_image_selection_module_results)
    for each_query_images in new_gui_candidate_image_selection_module_results:
        for each_img_dict in each_query_images:
            dataset_path = source_and_location_image_datasets[each_img_dict['Source']]
            each_img_dict['Image'] = dataset_path + each_img_dict['Image']
    
    ## create the final data structure and return it
    built_data_for_img_cap_no_attn_functionality = list()
    for query_key_elements_info, query_images_selected_info in zip(_id_elements_module_results, new_gui_candidate_image_selection_module_results):
        new_entry = dict()
        new_entry['key_elements'] = copy.deepcopy(query_key_elements_info)
        new_entry['selected_images_no_attn'] = list()
        for each_img_info_dict in query_images_selected_info:
            new_entry['selected_images_no_attn'].append( [each_img_info_dict['Image'] , None] )
        built_data_for_img_cap_no_attn_functionality.append(new_entry)
    return built_data_for_img_cap_no_attn_functionality

def img_cap_no_attn_inference_functionality(_id_elements_module_results, _gui_candidate_image_selection_module_results, _imgcap_inference_beam_width, _DEBUG_SWITCH):
    """
    ### TEMP CODE FOR TESTING - START - these variables will be passed directly to functionality in the combined logic script

    DEBUG_SWITCH = False
    ## data from ID Key Elements selection stage
    _id_elements_module_results = [[], ['car'], ['person', 'truck']]

    ## data from GUI Selection of Images returned by Neo4j images
    _gui_candidate_image_selection_module_results = [[], [{'Image': '000000033825.jpg', 'Source': 'coco_test_2017'}, {'Image': '000000155796.jpg', 'Source': 'coco_test_2017'}, {'Image': '000000224207.jpg', 'Source': 'coco_test_2017'}, {'Image': '000000313777.jpg', 'Source': 'coco_test_2017'}], [{'Image': '000000169542.jpg', 'Source': 'coco_test_2017'}, {'Image': '000000449668.jpg', 'Source': 'coco_test_2017'}, {'Image': '000000518174.jpg', 'Source': 'coco_test_2017'}, {'Image': '000000361201.jpg', 'Source': 'coco_test_2017'}, {'Image': '000000304424.jpg', 'Source': 'coco_test_2017'}]]
    
    # process command line arguments
    IMG_TO_INFER        = args.image_input            # -inimg parameter
    SAVED_WEIGHTS_PATH  = args.weights_file_input     # -inwtf parameter

    ### TEMP CODE FOR TESTING - END - these variables will be passed directly to functionality in the combined logic script
    """

    SAVED_WEIGHTS_PATH = r'/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Code/ModelsRuns/ImgCap/SavedData/Thesis_ImgCap_Weights_In_Run3/Decoder_Run_3_Wt_ep_18.h5'
    I2W_FILE = r'/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Code/ModelsRuns/ImgCap/SavedData/run2_1_kagg/ixtoword_train_97000.pkl'
    W2I_FILE = r'/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Code/ModelsRuns/ImgCap/SavedData/run2_1_kagg/wordtoix_train_97000.pkl'

    ## basic sanity checks
    if (not os.path.exists(I2W_FILE) ) or (not os.path.isfile(I2W_FILE )):
        myStr = "\n".join([
            f"\nFATAL ERROR: Index to Word dict not found at :: {I2W_FILE}",
            f"\nPremature Exit with exit code 110\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(110)
    if (not os.path.exists(W2I_FILE) ) or (not os.path.isfile(W2I_FILE )):
        myStr = "\n".join([
            f"\nFATAL ERROR: Word to Index dict not found at :: {W2I_FILE}",
            f"\nPremature Exit with exit code 111\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(111)
    if (not os.path.exists(SAVED_WEIGHTS_PATH) ) or (not os.path.isfile(SAVED_WEIGHTS_PATH )):
        myStr = "\n".join([
            f"\nFATAL ERROR: Weights file for Decoder model not found :: {SAVED_WEIGHTS_PATH}",
            f"\nPremature Exit with exit code 112\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(112)
    
    try:
        ### Load pre-trained model of Inception-v3 pretrained on Imagenet
        model_inception_v3_pretrained_imagement = tf.keras.applications.InceptionV3(weights='imagenet')
        # Create new model, by removing last layer (output layer) from Inception-V3
        model_CNN_encoder = keras.Model(inputs=model_inception_v3_pretrained_imagement.input, outputs=model_inception_v3_pretrained_imagement.layers[-2].output)
        ## only for DEBUG
        #type(model_CNN_encoder) ## should be tensorflow.python.keras.engine.functional.Functional
    except Exception as error_encoder_load_msg:
        myStr = "\n".join([
            f"\nFATAL ERROR: Could not load pretrained CNN-Encoder Inception-v3 model",
            f"\nError message :: {error_encoder_load_msg}\nPremature Exit with exit code 120\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(120)
    
    try:
        ## load the ixtoword and wordtoix dicts
        with open(I2W_FILE, 'rb') as handle:
            ixtoword = pickle.load(handle)
        with open(W2I_FILE, 'rb') as handle:
            wordtoix = pickle.load(handle)
        ## only for DEBUG
        if _DEBUG_SWITCH:
            print(f"Check wordtoix entries ::\nstartseq = {wordtoix.get('startseq')}\tendseq = {wordtoix.get('endseq')}\tbird = {wordtoix.get('bird')}")
            print(f"Check ixtoword entries ::\nix 1 = {ixtoword.get('1')}\tix 10 = {ixtoword.get('10')}\tix 1362 = {ixtoword.get('1362')}")
    except FileNotFoundError:
        myStr = "\n".join([
            f"\nFATAL ERROR: Could not load the Word2Ix and/ or Ix2Word dicts from specified locations ::",
            f"\nWord2Ix :: {W2I_FILE}\nIx2Word :: {I2W_FILE}",
            f"\nPremature Exit with exit code 125\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(125)
    except Exception as error_word_index_dict_load_msg:
        myStr = "\n".join([
            f"\nFATAL ERROR: unknown error loading the Word2Ix and/ or Ix2Word dicts :: {error_word_index_dict_load_msg}",
            f"\nPremature Exit with exit code 127\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(127)
    
    ## lengths of the the two dicts MUST be equal
    if len(wordtoix) != len(ixtoword):
        myStr = "\n".join([
            f"\nFATAL ERROR: Lengths of Word2Ix and Ix2Word dicts were not equal.",
            f"\nWord2Ix length = {len(wordtoix)}\t\t Ix2Word length = {len(ixtoword)}",
            f"\nPremature Exit with exit code 130\n"
            ])
        print_and_log(myStr, "error")
        exit(130)
    
    try:
        ### Load the Decoder network using the saved weights file
        ## Parameters to use while defining the Decoder model again
        EMBEDDING_DIMS = 200
        VOCAB_SIZE = 6758
        MAX_LENGTH_CAPTION = 49
        reloaded_RNN_decoder = reload_rnn_encoder_saved_weights(SAVED_WEIGHTS_PATH, EMBEDDING_DIMS, VOCAB_SIZE, MAX_LENGTH_CAPTION, _DEBUG_SWITCH)
        
        myStr = "\n".join([
            f"\nReloaded Image Captioning Decoder using weights file =",
            f"\n{SAVED_WEIGHTS_PATH}\n"
            ])
        print_and_log(myStr, "info")
        myStr = None

        if _DEBUG_SWITCH:
            type(reloaded_RNN_decoder)   ## should say     tensorflow.python.keras.engine.training.Model
    except Exception as error_decoder_load_msg:
        myStr = "\n".join([
            f"\nFATAL ERROR: Could not load LSTM-Decoder model.",
            f"Error message :: {error_decoder_load_msg}",
            f"\nPremature Exit with exit code 120\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(120)
    
    ## prepare the data for this functionality using data from earlier two stages
    try:
        data_for_img_cap_no_attn_functionality = convert_prev_stages_data(_id_elements_module_results, _gui_candidate_image_selection_module_results)
        if _DEBUG_SWITCH:
            print(f"\n\ndata_for_img_cap_no_attn_functionality =\n{data_for_img_cap_no_attn_functionality}\n\n")
    except Exception as data_prep_this_functionality_msg:
        myStr = "\n".join([
            f"\nFATAL ERROR: Problem preparing the data for this functionality using the data passed by previous stage.",
            f"\nData sent by previous stages:",
            f"\n _id_elements_module_results =",
            f"\n{_id_elements_module_results}",
            f"\n _gui_candidate_image_selection_module_results =",
            f"\n{_gui_candidate_image_selection_module_results}\n",
            f"\nPremature Exit with exit code 130\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(130)
    
    if _DEBUG_SWITCH:
        print(f"\n\nBEFORE:\n{data_for_img_cap_no_attn_functionality}\n\n")
    
    ## perform the inference and update the caption in the data structure
    for sentence_info in data_for_img_cap_no_attn_functionality:
        for each_img_info in sentence_info['selected_images_no_attn']:
            IMG_TO_INFER = each_img_info[0]
            out_caption = infer_for_caption_one_image(IMG_TO_INFER, MAX_LENGTH_CAPTION, model_CNN_encoder, reloaded_RNN_decoder, _imgcap_inference_beam_width, wordtoix, ixtoword, _DEBUG_SWITCH)
            ## strip off the startseq and endseq - note that startseq is ALWAYS present as the first word, but the endseq MAY NOT ALWAYS be at the end
            out_caption = out_caption.split(' ')
            if out_caption[-1] == 'endseq':
                out_caption = ' '.join(out_caption[1:-1]) ## remove both ends
            else:
                out_caption = ' '.join(out_caption[1:])   ## remove only startseq
            if _DEBUG_SWITCH:
                print(f"Inference caption = {out_caption}")
            each_img_info[1] = out_caption
    if _DEBUG_SWITCH:
        print(f"\n\nAFTER:\n{data_for_img_cap_no_attn_functionality}\n\n")
    
    img_cap_rc = 0
    return (img_cap_rc, None, data_for_img_cap_no_attn_functionality)

###############################################################################################################
## -------   IMAGE CAPTONING - NO ATTN - INFERENCE ENDS      IMAGE CAPTONING - NO ATTN - INFERENCE ENDS -------
###############################################################################################################

## ----------------------------------------------------------------------------------------------------------------------

###############################################################################################################
## ---- IMAGE CAPTONING - WITH ATTN - INFERENCE STARTS      IMAGE CAPTONING - WITH ATTN - INFERENCE STARTS ----
###############################################################################################################

def extend_data_structure_for_with_attn(_img_cap_after_no_attn_inference_module_results):
    """
    ## Add entries to hold the predictions of the WITH-attenion model
    ## -----------------------------------------------------------------------------
    example_in_data_structure = \
    [
        {
            'key_elements' : ['q1_obj1', 'q1_obj2'],
            'selected_images_no_attn' : [
                ['/path/to/q1_image1.jpg' , 'No_attn_model_caption_output_Img1'],
                ['/path/to/q1_image2.jpg' , 'No_attn_model_caption_output_Img2'],
                ['/path/to/q1_image3.jpg' , 'No_attn_model_caption_output_Img3']
            ]
        },
        {
            'key_elements' : [],
            'selected_images_no_attn' : []
        },
        {
            'key_elements' : ['q3_obj1', 'q3_obj2', 'q3_obj3'],
            'selected_images_no_attn' : []
        }
    ]

    #
    
    example_data_structure_to_be_prepared_by_this_function = \
    [
        {
            'key_elements' : ['q1_obj1', 'q1_obj2'],
            'selected_images_no_attn' : [
                ['/path/to/q1_image1.jpg' , 'No_attn_model_caption_output_Img1'],
                ['/path/to/q1_image2.jpg' , 'No_attn_model_caption_output_Img2'],
                ['/path/to/q1_image3.jpg' , 'No_attn_model_caption_output_Img3']
            ],
            'selected_images_with_attn' : [
                ['/path/to/q1_image1.jpg' , None],
                ['/path/to/q1_image2.jpg' , None],
                ['/path/to/q1_image3.jpg' , None]
            ]
        },
        {
            'key_elements' : [],
            'selected_images_no_attn' : [],
            'selected_images_with_attn' : []
        },
        {
            'key_elements' : ['q3_obj1', 'q3_obj2', 'q3_obj3'],
            'selected_images_no_attn' : [],
            'selected_images_with_attn' : []
        }
    ]
    ## -----------------------------------------------------------------------------
    """
    
    ## create the final data structure and return it
    updated_data_for_img_cap_functionality = copy.deepcopy(_img_cap_after_no_attn_inference_module_results)
    for sentence_info in updated_data_for_img_cap_functionality:
        sentence_info['selected_images_with_attn'] = list()
        for each_img_path_cap in sentence_info['selected_images_no_attn']:
            each_img_path , _ = each_img_path_cap
            sentence_info['selected_images_with_attn'].append( [each_img_path, None] )
    return updated_data_for_img_cap_functionality

def img_cap_WITH_attn_inference_functionality(_img_cap_after_no_attn_inference_module_results, _imgcap_inference_beam_width, _DEBUG_SWITCH):

    ## func defs - start
    class BahdanauAttention(tf.keras.Model):
        def __init__(self, units):
            super(BahdanauAttention, self).__init__()
            self.W1 = tf.keras.layers.Dense(units)
            self.W2 = tf.keras.layers.Dense(units)
            self.V = tf.keras.layers.Dense(1)

        def call(self, features, hidden):
            # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

            # hidden shape == (batch_size, hidden_size)
            # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
            hidden_with_time_axis = tf.expand_dims(hidden, 1)

            # attention_hidden_layer shape == (batch_size, 64, units)
            attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                                self.W2(hidden_with_time_axis)))

            # score shape == (batch_size, 64, 1)
            # This gives you an unnormalized score for each image feature.
            score = self.V(attention_hidden_layer)

            # attention_weights shape == (batch_size, 64, 1)
            attention_weights = tf.nn.softmax(score, axis=1)

            # context_vector shape after sum == (batch_size, hidden_size)
            context_vector = attention_weights * features
            context_vector = tf.reduce_sum(context_vector, axis=1)

            return context_vector, attention_weights

    class CNN_Encoder(tf.keras.Model):
        # Since you have already extracted the features and dumped it using pickle
        # This encoder passes those features through a Fully connected layer
        def __init__(self, embedding_dim):
            super(CNN_Encoder, self).__init__()
            # shape after fc == (batch_size, 64, embedding_dim)
            self.fc = tf.keras.layers.Dense(embedding_dim)

        def call(self, x):
            x = self.fc(x)
            x = tf.nn.relu(x)
            return x

    class RNN_Decoder(tf.keras.Model):
        def __init__(self, embedding_dim, units, vocab_size):
            super(RNN_Decoder, self).__init__()
            self.units = units

            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.gru = tf.keras.layers.GRU(self.units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
            self.fc1 = tf.keras.layers.Dense(self.units)
            self.fc2 = tf.keras.layers.Dense(vocab_size)

            self.attention = BahdanauAttention(self.units)

        def call(self, x, features, hidden):
            # defining attention as a separate model
            context_vector, attention_weights = self.attention(features, hidden)

            # x shape after passing through embedding == (batch_size, 1, embedding_dim)
            x = self.embedding(x)

            # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

            # passing the concatenated vector to the GRU
            output, state = self.gru(x)

            # shape == (batch_size, max_length, hidden_size)
            x = self.fc1(output)

            # x shape == (batch_size * max_length, hidden_size)
            x = tf.reshape(x, (-1, x.shape[2]))

            # output shape == (batch_size * max_length, vocab)
            x = self.fc2(x)

            return x, state, attention_weights

        def reset_state(self, batch_size):
            return tf.zeros((batch_size, self.units))

    ## Attention model - encoder related load function
    def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path

    def attn_populate_dummy_ckhpt_dir(_dest_chk_pt, _src_chk_pt_dir):
        dest_chk_pt = _dest_chk_pt[:]
        if dest_chk_pt[-1] != '/':
            dest_chk_pt = dest_chk_pt + r'/'
            
        count_copied = 0
        ## empty dest first
        for f in os.listdir(_dest_chk_pt):
            os.remove(_dest_chk_pt + f)
        
        myStr = f"Cleared current contents of the Destination dir\n"
        print_and_log(myStr, "info")
        myStr = None
        
        for f_src in os.listdir(_src_chk_pt_dir):
            orig_f = _src_chk_pt_dir + f_src
            tgt_f   = dest_chk_pt + f_src
            shutil.copyfile(orig_f, tgt_f)
            count_copied += 1
        
        myStr = f"Copied {count_copied} files\n\n\tFROM:\n{_src_chk_pt_dir}\n\tTO:\n{dest_chk_pt}\n"
        print_and_log(myStr, "info")
        myStr = None

    def attn_plot_attention(image, result, attention_plot):
        temp_image = np.array(Image.open(image))

        fig = plt.figure(figsize=(10, 10))

        len_result = len(result)
        for l in range(len_result):
            temp_att = np.resize(attention_plot[l], (8, 8))
            ax = fig.add_subplot(len_result//2, len_result//2, l+1)
            ax.set_title(result[l])
            img = ax.imshow(temp_image)
            ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

        plt.tight_layout()
        plt.show()

    def attn_evaluate_my_greedy(image):
        embedding_dim = 256
        units = 512
        top_k = 5000
        vocab_size = top_k + 1
        max_length = 52
        # Shape of the vector extracted from InceptionV3 is (64, 2048)
        # These two variables represent that vector shape
        features_shape = 2048
        attention_features_shape = 64
        
        attention_plot = np.zeros((max_length, attention_features_shape))

        hidden = decoder.reset_state(batch_size=1)

        temp_input = tf.expand_dims(load_image(image)[0], 0)
        img_tensor_val = image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        features = encoder(img_tensor_val)

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(max_length):
            predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

            #attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
            #predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            predicted_id = tf.math.argmax(predictions[0]).numpy()
            result.append(tokenizer.index_word[predicted_id])

            if tokenizer.index_word[predicted_id] == '<end>':
                return result, attention_plot

            dec_input = tf.expand_dims([predicted_id], 0)

        #attention_plot = attn_attention_plot[:len(result), :]
        attention_plot = None
        return result, attention_plot
    ## func defs -end

    ## tokenizer used while training the model
    TOKENIZER_PKL_FILE = r'/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Code/ModelsRuns/ImgCapAttention/SavedData/Thesis_ImgCapATTENTION_Deterministic_Run4/tokenizer_run4_from_training_100k.pkl'
    ## checkpoint dir for the inference
    CHKPT_DIR_OUT_ATTN = r'/home/rohit/PyWDUbuntu/thesis/ImgCapATTEND_opdir_DUMMY_RUN/chkPointDir_DUMMY_RUN/train_DUMMY_RUN/'
    ## checkpoint dir with files saved after training completed - used as input
    CHKPTS_IN_DIR_ATTN = r'/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Code/ModelsRuns/ImgCapAttention/SavedData/Thesis_ImgCapATTENTION_ChkPts_In_Run4_Ep21/'

    ## basic sanity checks
    if not os.path.exists(TOKENIZER_PKL_FILE):
        myStr = "\n".join([
            f"\nFATAL ERROR: Tokenizer pickle file for attention model not found here :: {TOKENIZER_PKL_FILE}",
            f"\nPremature Exit with exit code 110\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(110)
    if not os.path.exists(CHKPT_DIR_OUT_ATTN):
        myStr = "\n".join([
            f"\nFATAL ERROR: Checkpoint directory for run not found here :: {CHKPT_DIR_OUT_ATTN}",
            f"\nPremature Exit with exit code 111\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(111)
    if not os.path.exists(CHKPTS_IN_DIR_ATTN):
        myStr = "\n".join([
            f"\nFATAL ERROR: Checkpoints directory for saved checkpoint files reloading not found here :: {CHKPTS_IN_DIR_ATTN}",
            f"\nPremature Exit with exit code 112\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(112)
    
    ## define the attention decoder model parameters
    embedding_dim = 256
    units = 512
    top_k = 5000
    vocab_size = top_k + 1
    max_length = 52
    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    # These two variables represent that vector shape
    features_shape = 2048
    attention_features_shape = 64
    
    try:
        encoder = CNN_Encoder(embedding_dim)
        decoder = RNN_Decoder(embedding_dim, units, vocab_size)
        optimizer = tf.keras.optimizers.Adam()
    except Exception as error_attn_model_object_creation_msg:
        myStr = "\n".join([
            f"\nFATAL ERROR: Could not setup the Attention model objects",
            f"\nError message :: {error_attn_model_object_creation_msg}\nPremature Exit with exit code 120\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(120)
    
    try:
        ## load tokenizer from picked file
        with open(TOKENIZER_PKL_FILE, 'rb') as handle:
            tokenizer = pickle.load(handle)
    except FileNotFoundError:
        myStr = "\n".join([
            f"\nFATAL ERROR: Could not reload pickled tokenizer from here :: {TOKENIZER_PKL_FILE}",
            f"\nPremature Exit with exit code 125\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(125)
    except Exception as error_attn_tokenizer_reload_problem_msg:
        myStr = "\n".join([
            f"\nFATAL ERROR: Unknown error reloading tokenizer from here {TOKENIZER_PKL_FILE}",
            f"Error message :: {error_attn_tokenizer_reload_problem_msg}",
            f"\nPremature Exit with exit code 127\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(127)
    
    ## Initialize InceptionV3 and load the pretrained Imagenet weights
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    ## restore the checkpoint files to the run time directory
    DEST_DIR = CHKPT_DIR_OUT_ATTN
    SRC_DIR = CHKPTS_IN_DIR_ATTN
    attn_populate_dummy_ckhpt_dir(DEST_DIR, SRC_DIR)
    ## raise exception if the files are not the expected files
    expected_chkpt_dir_contents = [
        'ckpt-18.index',
        'ckpt-20.index',
        'checkpoint',
        'ckpt-19.data-00000-of-00001',
        'ckpt-21.data-00000-of-00001',
        'ckpt-21.index',
        'ckpt-22.index',
        'ckpt-22.data-00000-of-00001',
        'ckpt-19.index',
        'ckpt-20.data-00000-of-00001',
        'ckpt-18.data-00000-of-00001'
    ]
    if sorted(expected_chkpt_dir_contents) != sorted(os.listdir(CHKPT_DIR_OUT_ATTN)):
        problem_msg = '\n'.join([
            f"Checkpoint folder NOT CORRECTLY SETUP - Unable to restore the trained model\n",
            f"Expected contents =\n{sorted(expected_chkpt_dir_contents)}",
            f"Actual   contents =\n{sorted(os.listdir(CHKPT_DIR_OUT_ATTN))}"
        ])
        raise Exception(problem_msg)
    ## now actually restore the checkpoint
    checkpoint_path = CHKPT_DIR_OUT_ATTN
    ckpt = tf.train.Checkpoint(encoder=encoder,
                            decoder=decoder,
                            optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)
    
    ## extend data structure to handle the output of WITH-ATTENTION model
    data_for_img_cap_with_attn_functionality = extend_data_structure_for_with_attn(_img_cap_after_no_attn_inference_module_results)

    if _DEBUG_SWITCH:
        print(f"\n\nData structure NO-Attention: _img_cap_after_no_attn_inference_module_results = \n{_img_cap_after_no_attn_inference_module_results}\n")
        print(f"\nData structure WITH-Attention: data_for_img_cap_with_attn_functionality = \n{data_for_img_cap_with_attn_functionality}\n\n")
    
    ## perform the inference and update the caption in the data structure
    for sentence_info in data_for_img_cap_with_attn_functionality:
        for each_img_info in sentence_info['selected_images_with_attn']:
            IMG_TO_INFER , _ = each_img_info
            ## get predicted caption via greedy search
            out_caption, attention_plot = attn_evaluate_my_greedy(IMG_TO_INFER)
            ## remove endseq if present
            if out_caption[-1] == '<end>':
                out_caption = out_caption[:-1]
            out_caption = ' '.join(out_caption[:])
            if _DEBUG_SWITCH:
                print(f"Attention model Greedy search predicted caption for Image :: {IMG_TO_INFER} =\n{out_caption}\n")
            each_img_info[1] = out_caption
    if _DEBUG_SWITCH:
        print(f"\n\nAFTER:\n{data_for_img_cap_with_attn_functionality}\n\n")
    
    img_cap_rc = 0
    return (img_cap_rc, None, data_for_img_cap_with_attn_functionality)

###############################################################################################################
## -----   IMAGE CAPTONING - WITH ATTN - INFERENCE ENDS      IMAGE CAPTONING - WITH ATTN - INFERENCE ENDS -----
###############################################################################################################

## ----------------------------------------------------------------------------------------------------------------------

###############################################################################################################
## -----------   GUI DISPLAY IMAGE CAPTONING STARTS              GUI DISPLAY IMAGE CAPTONING STARTS -----------
###############################################################################################################

class c_imgCap_wnd_grid_window:
    def __init__(self, _root, _key_elements_info, _selected_imgs_info, _selected_imgs_info_attn, _query_num, _imgCap_deselected_positions, _DEBUG_SWITCH):
        self.root = _root
        self.key_elements_info = _key_elements_info
        self.selected_imgs_info = _selected_imgs_info
        self.selected_imgs_info_attn = _selected_imgs_info_attn
        self.query_num = _query_num
        self.imgCap_deselected_positions = _imgCap_deselected_positions
        self.DEBUG_SWITCH = _DEBUG_SWITCH

        ## how many images were sent previous stage
        self.num_images_in_this_query_at_start = len(self.selected_imgs_info)
        
        ## NOTE: This row count is only for  thumbnail image, path, caption, caption_attn, enlarge, edit buttons rows
        ##       NOT for Confirm button and Label for current count.
        ##           They will be referenced by using row count and indexing further down automatically.
        ## IMPORTANT: ENSURE #rows is multiple of 4 coz the logic in window grid array holding buttons/labels expects this
        ##
        ## 4 for image + 1 for Path + 1 for Enlarge + 1 for Edit caption + 1 for Edit caption_attn. NOTE: Path and Captions (both types) will span last two columns
        ## Logical Row:  first   row = 4 for image + 1 for path fixed label + 1 for path actual
        ## Logical Row:  second  row = 4 for image + 1 for caption fixed label + 1 for caption actual
        ## Logical Row:  third   row = 4 for image + 1 for caption_attn fixed label + 1 for caption_attn actual
        ## Logical Row:  fourth  row = 4 for image + 1 combined for Enlarge and/or Edit caption button
        ## This set of 4 rows for each logical set repeats for 5 possible images. Thus 4 x 5 = 20
        self.n_rows = 20  ## MUST BE DIVISIBLE BY 4
        ## 3 for image + 1 for path/caption fixed part + 1 for path/caption actual
        self.n_cols =  6
        
        ## array for the buttons for thumbnail image, path, caption, enlarge, edit buttons
        self.grid_buttons_arr = []
        ## The number of Selected candidate images cannot be more than this limit
        self.max_limit_images_selections = 5
        ## initialise the current count of selections to the number of images at start
        self.images_selected_current_count = self.num_images_in_this_query_at_start

        ## window for the grid selection
        self.wnd_grid = tk.Toplevel(master=self.root)
        self.wnd_grid.title(f"Thumbnails/Captions -- Query {self.query_num} - Key Elements -- {[kele for kele in self.key_elements_info]}")
        ## label for count of current selections
        self.lbl_track_selected_count = tk.Label(
            master=self.wnd_grid,
            text="",
            relief=tk.FLAT,
            borderwidth=10
            )
        ## call function to update the text with count, and the color 
        self.imgCap_update_label_count_currently_selected_images()
        ## button for selection confirm
        self.btn_confirm_selections = tk.Button(
            master=self.wnd_grid,
            text=f"Click to CONFIRM",
            bg="yellow", fg="black",
            borderwidth=10,
            relief=tk.RAISED,
            command=self.imgCap_do_confirm_selections_processing
            )
        self.btn_confirm_selections.configure(
            width=( len(self.btn_confirm_selections["text"]) + 20 ),
            height=5
            )
        ## prevent Confirm Selections if there are too many images Selected at start
        if self.images_selected_current_count > self.max_limit_images_selections:
            self.btn_confirm_selections.configure(state=tk.DISABLED, relief=tk.FLAT)
        
        ## populate the 2-D button array for the grid -  thumbnail image, path, caption, enlarge, edit buttons ONLY
        ##    first make skeleton entries for the buttons
        ##    by default assume no Image is present to display,
        ##       so all buttons are in disabled state with:
        ##           Image         label           = "No Image"
        ##           Path          label fixed     = "Path:"
        ##           Path          label           = "No Path"
        ##           Caption       label fixed     = "Caption:"
        ##           Caption       label           = "No Caption"
        ##           Caption_Attn  label fixed     = "Caption_Attn:"
        ##           Caption_Attn  label           = "No Caption"
        for r_idx in range(self.n_rows):
            self.wnd_grid.rowconfigure(r_idx, weight=1, minsize=20)
            ## image will span three rows
            if r_idx % 4 == 0:
                temp_row_data = []
                ## thumbnail image button - pos 0 in inner list
                temp_row_data.append(
                    tk.Button(
                        master=self.wnd_grid,
                        text="No Image",
                        bg="black", fg="white",
                        relief=tk.FLAT,
                        borderwidth=10,
                        state=tk.DISABLED
                        ))
                ## associated Path label FIXED value - pos 1 in inner list
                temp_row_data.append(
                    tk.Label(
                        master=self.wnd_grid,
                        text=" Path: ",
                        bg="grey", fg="white",
                        relief=tk.SUNKEN,
                        borderwidth=5
                        ))
                ## associated Path label ACTUAL value - pos 2 in inner list
                temp_row_data.append(
                    tk.Label(
                        master=self.wnd_grid,
                        text="No Path",
                        bg="black", fg="white",
                        relief=tk.FLAT,
                        borderwidth=5
                        ))
                ## associated caption label FIXED - pos 3 in inner list
                temp_row_data.append(
                    tk.Label(
                        master=self.wnd_grid,
                        text=" Caption: ",
                        bg="grey", fg="white",
                        relief=tk.SUNKEN,
                        borderwidth=5
                        ))
                ## associated caption ACTUAL label - pos 4 in inner list
                temp_row_data.append(
                    tk.Label(
                        master=self.wnd_grid,
                        text="No Caption",
                        bg="black", fg="white",
                        relief=tk.FLAT,
                        borderwidth=5
                        ))
                ## associated caption_attn label FIXED - pos 5 in inner list
                temp_row_data.append(
                    tk.Label(
                        master=self.wnd_grid,
                        text=" Caption_Attn: ",
                        bg="grey", fg="white",
                        relief=tk.SUNKEN,
                        borderwidth=5
                        ))
                ## associated caption_attn ACTUAL label - pos 6 in inner list
                temp_row_data.append(
                    tk.Label(
                        master=self.wnd_grid,
                        text="No Caption",
                        bg="black", fg="white",
                        relief=tk.FLAT,
                        borderwidth=5
                        ))
                ## associated enlarge and/or edit caption button - pos 7 in inner list
                temp_row_data.append(
                    tk.Button(
                        master=self.wnd_grid,
                        text="DISABLED: Click to Enlarge Image and/or Edit Caption",
                        bg="black", fg="white",
                        relief=tk.SUNKEN,
                        borderwidth=5,
                        state=tk.DISABLED
                        ))
                self.grid_buttons_arr.append(temp_row_data)
            for c_idx in range(self.n_cols):
                self.wnd_grid.columnconfigure(c_idx, weight=1, minsize=20)
        
        if self.DEBUG_SWITCH:
            print(f"shape of grid_buttons_arr = ({len(self.grid_buttons_arr)} , {len(self.grid_buttons_arr[0])})")
        
        ## set grid positions for all the buttons and labels just created:
        ##     thumbnail button, path label, caption label, caption_attn label, enlarge/edit caption button
        ## NOTE: going from the button grid 2-d list and mapping to actual grid positions
        for r_idx, logical_set_for_image in enumerate(self.grid_buttons_arr):
            for c_idx, button_or_label in enumerate(logical_set_for_image):
                ## c_idx              type
                ## 0                  image thumbnail button
                ## 1                  path FIXED label
                ## 2                  path ACTUAL label
                ## 3                  caption FIXED label
                ## 4                  caption ACTUAL label
                ## 5                  caption_attn FIXED label
                ## 6                  caption_attn ACTUAL label
                ## 7                  enlarge and/or edit caption button
                if c_idx == 0:
                    ## image button
                    button_or_label.grid(
                        row=r_idx * 4, column=0,
                        rowspan=4, columnspan=4,
                        padx=3, pady=3,
                        sticky="nsew")
                elif c_idx == 1:
                    ## path FIXED label
                    button_or_label.grid(
                        row=r_idx * 4, column=5,
                        rowspan=1, columnspan=1,
                        padx=3, pady=3,
                        sticky="nsew")
                elif c_idx == 2:
                    ## path ACTUAL label
                    button_or_label.grid(
                        row=r_idx * 4, column=6,
                        rowspan=1, columnspan=1,
                        padx=3, pady=3,
                        sticky="nsew")
                elif c_idx == 3:
                    ## caption FIXED label
                    button_or_label.grid(
                        row=r_idx * 4 + 1, column=5,
                        rowspan=1, columnspan=1,
                        padx=3, pady=3,
                        sticky="nsew")
                elif c_idx == 4:
                    ## caption ACTUAL label
                    button_or_label.grid(
                        row=r_idx * 4 + 1, column=6,
                        rowspan=1, columnspan=1,
                        padx=3, pady=3,
                        sticky="nsew")
                elif c_idx == 5:
                    ## caption_attn FIXED label
                    button_or_label.grid(
                        row=r_idx * 4 + 2, column=5,
                        rowspan=1, columnspan=1,
                        padx=3, pady=3,
                        sticky="nsew")
                elif c_idx == 6:
                    ## caption ACTUAL label
                    button_or_label.grid(
                        row=r_idx * 4 + 2, column=6,
                        rowspan=1, columnspan=1,
                        padx=3, pady=3,
                        sticky="nsew")
                elif c_idx == 7:
                    ## enlarge and/or edit caption button
                    button_or_label.grid(
                        row=r_idx * 4 + 3, column=5,
                        rowspan=1, columnspan=2,
                        padx=3, pady=3,
                        sticky="nsew")
        
        for r_idx, logical_set_for_image in enumerate(self.grid_buttons_arr):
            if r_idx >= self.num_images_in_this_query_at_start:
                ## no more images present
                break
            for c_idx, button_or_label in enumerate(logical_set_for_image):
                ## c_idx              type
                ## 0                  image thumbnail button
                ## 1                  path FIXED label
                ## 2                  path ACTUAL label
                ## 3                  caption FIXED label
                ## 4                  caption ACTUAL label
                ## 5                  caption_attn FIXED label
                ## 6                  caption_attn ACTUAL label
                ## 7                  enlarge and/or edit caption button
                if c_idx == 0:
                    ## image button
                    if self.DEBUG_SWITCH == True:
                        print(f"\nGonna try loading thumbnail for image =\n{self.selected_imgs_info[r_idx][0]}\n")
                    #img_orig = Image.open( self.selected_imgs_info[r_idx][0] ).resize((80, 80),Image.ANTIALIAS)
                    #img_resized = ImageTk.PhotoImage(img_orig)
                    img_resized = ImageTk.PhotoImage( Image.open( self.selected_imgs_info[r_idx][0] ).resize((120, 120),Image.ANTIALIAS) )
                    button_or_label.configure(
                            image=img_resized,
                            relief=tk.SUNKEN,
                            borderwidth=10,
                            highlightthickness = 15,
                            highlightbackground = "green", highlightcolor= "green",
                            state=tk.NORMAL,
                            command=partial(
                                self.imgCap_do_image_select_button_clicked_processing,
                                r_idx, c_idx,
                                self.DEBUG_SWITCH
                                )
                        )
                    button_or_label.image = img_resized
                elif c_idx == 2:
                    ## path ACTUAL label
                    button_or_label.configure(
                        text=self.selected_imgs_info[r_idx][0],
                        bg="grey", fg="white",
                        relief=tk.SUNKEN,
                        padx=3, pady=3)
                    button_or_label.configure(
                        width=( len(button_or_label["text"]) + 10 ),
                        height=3
                        )
                elif c_idx == 4:
                    ## caption ACTUAL label
                    button_or_label.configure(
                        text=self.selected_imgs_info[r_idx][1],
                        bg="blue", fg="white",
                        relief=tk.FLAT,
                        padx=3, pady=3)
                    button_or_label.configure(
                        width=( len(button_or_label["text"]) + 10 ),
                        height=3
                        )
                elif c_idx == 6:
                    ## caption_attn ACTUAL label
                    button_or_label.configure(
                        text=self.selected_imgs_info_attn[r_idx][1],
                        bg="blue", fg="white",
                        relief=tk.FLAT,
                        padx=3, pady=3)
                    button_or_label.configure(
                        width=( len(button_or_label["text"]) + 10 ),
                        height=3
                        )
                elif c_idx == 7:
                    ## enlarge and/or edit caption button
                    button_or_label.configure(
                        relief=tk.RAISED,
                        text="Click to Enlarge Image and/or Edit Caption",
                        bg="yellow", fg="black",
                        state=tk.NORMAL,
                        command=partial(
                            imgcap_generic_show_enlarge_image_edit_caption_window,
                            self,
                            r_idx, c_idx,
                            self.DEBUG_SWITCH
                        )
                        )
        
        ## label for count of current selections
        r_idx = self.n_rows
        c_idx = 0
        self.lbl_track_selected_count.grid(
            row=r_idx, column=c_idx,
            rowspan=1, columnspan=self.n_cols + 1,
            sticky="nsew",
            padx=5, pady=5
            )

        ## button for selection confirm
        r_idx = self.n_rows + 1
        c_idx = 0
        self.btn_confirm_selections.grid(
            row=r_idx, column=c_idx,
            rowspan=1, columnspan=self.n_cols + 1,
            sticky="nsew",
            padx=5, pady=5
            )
        
        return
        
    def imgCap_do_image_select_button_clicked_processing(self, _r_idx, _c_idx, _DEBUG_SWITCH):
        ## toggle button characteristics:
        ##                              Relief         Color around image
        ##       Selected   Image       SUNKEN         Green
        ##       Deselected Image       RAISED         Red
        if _DEBUG_SWITCH:
            print(f"\n\nImage button clicked :: (r_idx, c_idx) = {_r_idx, _c_idx}\n\n")
        ## toggle the button characteristics
        if self.grid_buttons_arr[_r_idx][_c_idx]["relief"] == tk.SUNKEN:
            ## was Selected, change to Deselected
            self.grid_buttons_arr[_r_idx][_c_idx].configure(
                relief=tk.RAISED,
                highlightbackground = "red", highlightcolor= "red"
                )
            self.images_selected_current_count -= 1
        else:
            ## was Deselected, change to Selected
            self.grid_buttons_arr[_r_idx][_c_idx].configure(
                relief=tk.SUNKEN,
                highlightbackground = "green", highlightcolor= "green"
                )
            self.images_selected_current_count += 1

        ## update the count of currently selected images
        self.imgCap_update_label_count_currently_selected_images()

    def imgCap_update_label_count_currently_selected_images(self):
        ## update the count based on latest count of selected images
        ##        also change the color if the count is greater than allowed limit
        self.lbl_track_selected_count.configure(
            text=" ".join([ "Count of Images currently Selected =", str(self.images_selected_current_count) ])
            )
        self.lbl_track_selected_count.configure(
            width=( len(self.lbl_track_selected_count["text"]) + 10 ),
            height=5
            )
        if self.images_selected_current_count > self.max_limit_images_selections:
            self.lbl_track_selected_count.configure(bg="red", fg="white")
        else:
            self.lbl_track_selected_count.configure(bg="green", fg="white")
        return
    
    def imgCap_do_confirm_selections_processing(self):
        ## For the images that are Deselected, figure out the position and add it to the deselections list.
        if self.DEBUG_SWITCH:
            print(f"\n\nCONFIRM button was pressed\n\n")
        for r_idx, logical_set_for_image in enumerate(self.grid_buttons_arr):
            img_button_info, _, _, _, _, _, _, _ = logical_set_for_image ## only interested in the first entry
            if img_button_info["relief"] == tk.RAISED:
                ## this image was deselected -- add to list
                self.imgCap_deselected_positions.append(r_idx)

        myStr = f"\nFor Query {self.query_num}, Deselected positions=\n{self.imgCap_deselected_positions}\n"
        print_and_log(myStr, "debug")
        myStr = None
        
        self.root.destroy()
        return

def imgcap_generic_show_enlarge_image_edit_caption_window(_o_wnd_grid, _r_idx, _c_idx, _DEBUG_SWITCH):
    if _DEBUG_SWITCH:
        print(f"\nEntered the enlarge n edit window for img captioning\n")
    ## show the new window for enlarge image with edit caption option
    _o_imgCap_enlargeEdit_window = c_imgCap_enlargeEdit_window(_o_wnd_grid, _r_idx, _c_idx, _DEBUG_SWITCH)
    _o_imgCap_enlargeEdit_window.wnd_enlarged_img.mainloop()
    return

def imgcap_generic_show_grid_selection_window(_root, _key_elements_info, _selected_imgs_info, _selected_imgs_info_attn, _query_num, _imgCap_deselected_positions, _DEBUG_SWITCH):
    if _DEBUG_SWITCH:
        print(f"\nEntered the show grid selection window for img captioning\n")
    
    o_imgCap_wnd_grid_window = c_imgCap_wnd_grid_window(_root, _key_elements_info, _selected_imgs_info, _selected_imgs_info_attn, _query_num, _imgCap_deselected_positions, _DEBUG_SWITCH)
    o_imgCap_wnd_grid_window.wnd_grid.mainloop()
    return

class c_imgCap_enlargeEdit_window():
    def __init__(self, _o_wnd_grid, _r_idx, _c_idx, _DEBUG_SWITCH):
        self.root = _o_wnd_grid.wnd_grid

        ## extract caption as string - will not be editable
        self.caption_orig = _o_wnd_grid.grid_buttons_arr[_r_idx][4]["text"]
        ## make caption copy that will be editable
        self.caption_editable = self.caption_orig
        ## extract caption_attn as string - will not be editable
        self.caption_attn_orig = _o_wnd_grid.grid_buttons_arr[_r_idx][6]["text"]
        ## make caption_attn copy that will be editable
        self.caption_attn_editable = self.caption_attn_orig
        
        self.img_to_enlarge_path = _o_wnd_grid.selected_imgs_info[_r_idx][0]
        self.query_num = _o_wnd_grid.query_num
        self.DEBUG_SWITCH = _DEBUG_SWITCH

        ## window for enlarged image
        self.wnd_enlarged_img = tk.Toplevel(master=self.root)
        self.wnd_enlarged_img.title(f"Enlarged Image and Optional Caption Editing -- {os.path.basename(self.img_to_enlarge_path)}")

        ## label for enlarged image
        self.lbl_enlarged_image = tk.Label(
            master=self.wnd_enlarged_img,
            image=None,
            relief=tk.FLAT,
            borderwidth=5
            )
        
        ## insert the enlarged image image
        img_enlarged = ImageTk.PhotoImage(
            Image.open(self.img_to_enlarge_path).resize((400, 400),Image.ANTIALIAS)
            )
        self.lbl_enlarged_image.configure(
            image=img_enlarged
            )
        self.lbl_enlarged_image.image = img_enlarged

        ## label for original caption
        self.lbl_orig_caption = tk.Label(
            master=self.wnd_enlarged_img,
            text="ORIGINAL CAPTION:: " + self.caption_orig,
            bg="grey", fg="white",
            relief=tk.SUNKEN,
            borderwidth=5
            )
        self.lbl_orig_caption.configure(
            width=( len(self.lbl_orig_caption["text"]) + 10 ),
            height=5
        )

        ## text widget for editable caption
        self.txt_editable_caption = tk.Text(
            master=self.wnd_enlarged_img,
            width = len(self.caption_editable) + 10,
            exportselection=0, ## otherwise only any accidentally selected text will be captured
            height=3,
            wrap=tk.WORD,
            state=tk.NORMAL
            )
        self.txt_editable_caption.insert(tk.END, self.caption_editable)

        ## label for original caption_attn
        self.lbl_orig_caption_attn = tk.Label(
            master=self.wnd_enlarged_img,
            text="ORIGINAL CAPTION_ATTN:: " + self.caption_attn_orig,
            bg="grey", fg="white",
            relief=tk.SUNKEN,
            borderwidth=5
            )
        self.lbl_orig_caption_attn.configure(
            width=( len(self.lbl_orig_caption_attn["text"]) + 10 ),
            height=5
        )

        ## text widget for editable caption_attn
        self.txt_editable_caption_attn = tk.Text(
            master=self.wnd_enlarged_img,
            width = len(self.caption_attn_editable) + 10,
            exportselection=0, ## otherwise only any accidentally selected text will be captured
            height=3,
            wrap=tk.WORD,
            state=tk.NORMAL
            )
        self.txt_editable_caption_attn.insert(tk.END, self.caption_attn_editable)
        
        ## label for caption editing instruction
        self.lbl_edit_caption_message = tk.Label(
            master=self.wnd_enlarged_img,
            text="Optional: You may EDIT either or both captions in the white box above and then click the CONFIRM button below",
            bg="grey", fg="white",
            relief=tk.SUNKEN,
            borderwidth=5
            )
        self.lbl_edit_caption_message.configure(
            width=( len(self.lbl_edit_caption_message["text"]) + 10 ),
            height=3
        )

        ## button to confirm caption editing done
        self.btn_confirm_enlarged_window_done = tk.Button(
            master=self.wnd_enlarged_img,
            text=f"Click to CONFIRM changes to caption (if any) and close this window",
            bg="yellow", fg="black",
            relief=tk.RAISED,
            borderwidth=15,
            command=partial(
                self.imgcap_generic_enlarge_edit_caption_window_confirm,
                _o_wnd_grid,
                _r_idx,
                self.DEBUG_SWITCH
                )
            )
        self.btn_confirm_enlarged_window_done.configure(
            width=( len(self.btn_confirm_enlarged_window_done["text"]) + 10 ),
            height=5
        )
        
        ## pack the widgets
        # self.imgcap_enlarge_edit_pack_items()
        self.lbl_enlarged_image.pack(padx=3, pady=3)
        self.lbl_edit_caption_message.pack(padx=3, pady=3)
        self.lbl_orig_caption.pack(padx=3, pady=3)
        self.txt_editable_caption.pack(padx=3, pady=3)
        self.lbl_orig_caption_attn.pack(padx=3, pady=3)
        self.txt_editable_caption_attn.pack(padx=3, pady=3)
        self.btn_confirm_enlarged_window_done.pack(padx=5, pady=5)
        return

    def imgcap_generic_enlarge_edit_caption_window_confirm(self, _o_wnd_grid, _r_idx, _DEBUG_SWITCH):
        if _DEBUG_SWITCH:
            print(f"\n\nConfirm button pressed on enlarged image window")
            print(f"\n\tORIG CAPTION=\n{self.caption_orig}\n\tEDITED CAPTION=\n{self.txt_editable_caption.get('1.0','end-1c')}\n")
            print(f"\n\tORIG CAPTION_ATTN=\n{self.caption_attn_orig}\n\tEDITED CAPTION_ATTN=\n{self.txt_editable_caption_attn.get('1.0','end-1c')}\n")
        
        ## capture the editable caption texts from widget
        self.caption_editable = self.txt_editable_caption.get('1.0','end-1c')
        self.caption_attn_editable = self.txt_editable_caption_attn.get('1.0','end-1c')

        if (self.caption_orig == self.caption_editable) and (self.caption_attn_orig == self.caption_attn_editable):
            myStr = "".join([
                f"\n\nImgCap:: After Enlarge window for Query {self.query_num},",
                f" Image=\n{self.img_to_enlarge_path}",
                f"\nCaption NOT CHANGED",
                f"\nOriginal Caption =\n{self.caption_orig}",
                f"\nEditable Caption =\n{self.caption_editable}\n",
                f"\nCaption_Attn NOT CHANGED",
                f"\nOriginal Caption_Attn =\n{self.caption_attn_orig}",
                f"\nEditable Caption_Attn =\n{self.caption_attn_editable}\n"
            ])
            print_and_log(myStr, "debug")
            myStr = None
        else:
            ## either one or both captions have been edited - check each one individually and action
            
            if self.caption_orig != self.caption_editable:
                ## normal caption edited
                myStr = "".join([
                    f"\n\nImgCap:: After Enlarge window for Query {self.query_num},",
                    f" Image=\n{self.img_to_enlarge_path}",
                    f"\nCaption WAS CHANGED",
                    f"\nOriginal Caption =\n{self.caption_orig}",
                    f"\nEditable Caption =\n{self.caption_editable}\n"
                ])
                print_and_log(myStr, "debug")
                myStr = None

                ## overwrite the original caption with new value and change color of the text in label
                _o_wnd_grid.grid_buttons_arr[_r_idx][4].configure(
                    text=self.caption_editable,
                    bg="yellow",fg="black"
                )
                _o_wnd_grid.grid_buttons_arr[_r_idx][4].configure(
                    width = len(_o_wnd_grid.grid_buttons_arr[_r_idx][4]["text"]) + 30,
                    bg="yellow",fg="black"
                )
                ## also overwrite the data structure entry
                _o_wnd_grid.selected_imgs_info[_r_idx][1] = self.caption_editable
            
            if self.caption_attn_orig != self.caption_attn_editable:
                ## attention caption edited
                myStr = "".join([
                    f"\n\nImgCap:: After Enlarge window for Query {self.query_num},",
                    f" Image=\n{self.img_to_enlarge_path}",
                    f"\nCaption_Attn WAS CHANGED",
                    f"\nOriginal Caption_Attn =\n{self.caption_attn_orig}",
                    f"\nEditable Caption_Attn =\n{self.caption_attn_editable}\n"
                ])
                print_and_log(myStr, "debug")
                myStr = None

                ## overwrite the original caption_attn with new value and change color of the text in label
                _o_wnd_grid.grid_buttons_arr[_r_idx][6].configure(
                    text=self.caption_attn_editable,
                    bg="yellow",fg="black"
                )
                _o_wnd_grid.grid_buttons_arr[_r_idx][6].configure(
                    width = len(_o_wnd_grid.grid_buttons_arr[_r_idx][6]["text"]) + 30,
                    bg="yellow",fg="black"
                )
                ## also overwrite the data structure entry
                _o_wnd_grid.selected_imgs_info_attn[_r_idx][1] = self.caption_attn_editable
        
        ## destroy enlarged image window
        self.wnd_enlarged_img.destroy()
        
        return

class c_queryImgCapGuiOneQuery():
    def __init__(self, _DEBUG_SWITCH, _query_num, _key_elements_info, _selected_imgs_info, _selected_imgs_info_attn, _imgCap_deselected_positions):
        self.DEBUG_SWITCH = _DEBUG_SWITCH
        self.query_num = _query_num
        self.key_elements_info = _key_elements_info
        self.selected_imgs_info = _selected_imgs_info
        self.selected_imgs_info_attn = _selected_imgs_info_attn
        self.count_imgs_to_display = len(_selected_imgs_info)
        self.imgCap_deselected_positions = _imgCap_deselected_positions
        if _DEBUG_SWITCH:
            print(f"\n\nself.count_imgs_to_display = {self.count_imgs_to_display}\n\n")
        
        self.msg_warning_for_root_do_not_close = r'DO NOT CLOSE THIS WINDOW TILL YOU ARE READY TO MOVE TO THE NEXT QUERY!!!'
        self.msg_instructions_for_root = "".join([
            f"                               ----------------------------------",
            f"\n                               -----      INSTRUCTIONS      -----",
            f"\n                               ----------------------------------",
            f"\n",
            f"\n",
            f"\nThis is the main control window for the results of a particular query.",
            f"\n",
            f"\n\t1.  Click on the button to proceed to display a grid of images and their captions.",
            f"\n\t       Important: If there are no images to display you cannot click to proceed to the"
            f"\n\t                  grid selection window. Simply close this window and the next query"
            f"\n\t                  selections process will begin."
            f"\n",
            f"\n\t2.  On the grid selection window, you can click the ENLARGE/EDIT button to see an enlarged image.",
            f"\n\t       Optional: There you may edit either or both captions.",
            f"\n\t       Once caption is edited, click the CONFIRM button to save the changes.",
            f"\n",
            f"\n\t3.  Once you are ready to move on to the next query results, click the CONFIRM button on the grid selection window.",
            f"\n"
        ])

        ## root window for the query being processed
        self.root = tk.Tk()
        if self.count_imgs_to_display == 0:
            self.root.title(f"Image Captions - Query number {self.query_num} - No Images to Display -- Please close this window")
        else:
            self.root.title(f"Image Captions - Query number {self.query_num} - Click button to Proceed")
        ## label for warning message not to close
        self.lbl_root_msg_warning_not_close = tk.Label(
            master=self.root,
            text=self.msg_warning_for_root_do_not_close,
            bg="red", fg="white",
            width=(len(self.msg_warning_for_root_do_not_close) + 10),
            height=5
            )
        ## label for instructions
        self.lbl_root_instructions = tk.Label(
            master=self.root,
            text=self.msg_instructions_for_root,
            bg="blue", fg="white",
            justify=tk.LEFT,
            width=130,
            height=25,
            relief=tk.SUNKEN
            )
        ## button to proceed to grid selection window
        ## assume there are images and make it clickable
        self.btn_root_click_proceed = tk.Button(
            master=self.root,
            text=f"Click to view images and captions - Query {self.query_num}",
            bg="yellow", fg="black",
            relief=tk.RAISED,
            borderwidth=15,
            command=partial(
                imgcap_generic_show_grid_selection_window,
                self.root,
                self.key_elements_info,
                self.selected_imgs_info,
                self.selected_imgs_info_attn,
                self.query_num,
                self.imgCap_deselected_positions,
                self.DEBUG_SWITCH
                )
            )
        ## if no images to display in grid, disable the button to proceed and change the text displayed in button
        if self.count_imgs_to_display == 0:
            self.btn_root_click_proceed.configure(
            state=tk.DISABLED,
            relief=tk.FLAT,
            text=f"No images for this - Query {_query_num} - Please close this window",
            )
        self.btn_root_click_proceed.configure(
            width=(len(self.btn_root_click_proceed["text"]) + 10),
            height=7
        )
        
        self.lbl_root_msg_warning_not_close.pack(padx=15, pady=15)
        self.lbl_root_instructions.pack(padx=10, pady=10)
        self.btn_root_click_proceed.pack(padx=50, pady=50)


def gui_img_cap_functionality_for_one_query(_DEBUG_SWITCH, _query_num, _key_elements_info, _selected_imgs_info, _selected_imgs_info_attn, _imgCap_index_positions_to_remove_this_query):
    if _DEBUG_SWITCH:
        print(f"\n\nquery = {_query_num}\nkeyEleInfo = {_key_elements_info}\nselecImgInfoNoAttn = {_selected_imgs_info}\nselecImgInfoWithAttn = {_selected_imgs_info_attn}\n\n")
    ## make a root window and show it
    o_queryImgCapGuiOneQuery_root_window = c_queryImgCapGuiOneQuery(_DEBUG_SWITCH, _query_num, _key_elements_info, _selected_imgs_info, _selected_imgs_info_attn, _imgCap_index_positions_to_remove_this_query)
    o_queryImgCapGuiOneQuery_root_window.root.mainloop()

def gui_img_cap_show_gui_functionality(_img_cap_inference_module_results, _file4stygen, _DEBUG_SWITCH):
    if _DEBUG_SWITCH:
        print(f"\nDATA IN img_cap_inference_module_results=\n{_img_cap_inference_module_results}\n")
    
    ## made copy of the info passed from prev stage, intent is to let user edit the captions in this copy
    copy_img_cap_inference_module_results = copy.deepcopy(_img_cap_inference_module_results)
    
    imgCap_index_positions_to_remove_all_queries = []
    for query_num, query_img_cap_results in enumerate(copy_img_cap_inference_module_results):
        num_imgCap_candidate_images_before_selection_began = len(query_img_cap_results['selected_images_no_attn'])
        imgCap_index_positions_to_remove_this_query = []
        gui_img_cap_functionality_for_one_query(_DEBUG_SWITCH, query_num+1, query_img_cap_results['key_elements'], query_img_cap_results['selected_images_no_attn'], query_img_cap_results['selected_images_with_attn'], imgCap_index_positions_to_remove_this_query)
        imgCap_index_positions_to_remove_all_queries.append(imgCap_index_positions_to_remove_this_query)

        num_images_to_remove = len(imgCap_index_positions_to_remove_this_query)
        myStr = "\n".join([
            f"\nCompleted selection process - Query number {query_num + 1}",
            f"Deselected image positions this query = {imgCap_index_positions_to_remove_this_query}",
            f"Number of images before selection began = {num_imgCap_candidate_images_before_selection_began}",
            f"Number of images Deselected by user = {num_images_to_remove}.",
            f"Number of images that will remain = { num_imgCap_candidate_images_before_selection_began - num_images_to_remove }\n"
            ])
        print_and_log(myStr, "debug")
        myStr = None
    
    myStr = "\n".join([
        f"\n\n\n******************************************************************",
        f"********* SUMMARY for Image Captioning GUI - all queries *********",
        f"*********     BEFORE removing the deselected images      *********",
        f"******************************************************************",
        f"Data structure BEFORE =\n{_img_cap_inference_module_results}",
        f"Data structure AFTER GUI only; only changed captions should reflect but no deselection done as yet  =\n{copy_img_cap_inference_module_results}\n"
        ])
    print_and_log(myStr, "debug")
    myStr = None

    ## remove the deselected images from data structure
    final_img_cap_inference_module_results=[]
    for each_query_result, each_query_pos_remove_list in zip(copy_img_cap_inference_module_results, imgCap_index_positions_to_remove_all_queries):
        final_img_cap_inference_module_results.append({'key_elements': [], 'selected_images_no_attn': [], 'selected_images_with_attn': []})
        final_img_cap_inference_module_results[-1]['key_elements'] = each_query_result['key_elements']
        final_img_cap_inference_module_results[-1]['selected_images_no_attn'] = [val for idx, val in enumerate(each_query_result['selected_images_no_attn']) if idx not in each_query_pos_remove_list]
        final_img_cap_inference_module_results[-1]['selected_images_with_attn'] = [val for idx, val in enumerate(each_query_result['selected_images_with_attn']) if idx not in each_query_pos_remove_list]
    
    myStr = "\n".join([
        f"\n\n\n******************************************************************",
        f"********* SUMMARY for Image Captioning GUI - all queries *********",
        f"*********     AFTER removing the deselected images      *********",
        f"******************************************************************",
        f"Data structure BEFORE =\n{_img_cap_inference_module_results}",
        f"Deselected image positions all queries = {imgCap_index_positions_to_remove_all_queries}",
        f"Data structure AFTER GUI; changed captions and removing Deselected images =\n{final_img_cap_inference_module_results}\n"
        ])
    print_and_log(myStr, "debug")
    myStr = None

    ## save to file for the next stage to pick up later
    try:
        with open(_file4stygen, "w") as f:
            json.dump(final_img_cap_inference_module_results, f)
        myStr = "\n".join([
            f"\nSaved results to file for the Story Generator stage.",
            f"File :: {_file4stygen}"
            ])
        print_and_log(myStr, "info")
        myStr = None
    except Exception as imgCap_post_gui_data_save_error_msg:
        myStr = "\n".join([
            f"\nFATAL ERROR: Problem saving data to file for the Story Generator stage.",
            f"Error message :: {imgCap_post_gui_data_save_error_msg}",
            f"Tried to save to file location here =\n{_file4stygen}"
            ])
        print_and_log(myStr, "error")
        return 100, myStr, None

    return (0, None, final_img_cap_inference_module_results)

###############################################################################################################
## -----------     GUI DISPLAY IMAGE CAPTONING ENDS              GUI DISPLAY IMAGE CAPTONING ENDS   -----------
###############################################################################################################

## ----------------------------------------------------------------------------------------------------------------------

###############################################################################################################
## ----------------------  CONTROL LOGIC STARTS                    CONTROL LOGIC STARTS ---------------------- 
###############################################################################################################
def print_and_log(_in_fstring_msg, _log_level):
    print(f"LOG_LEVEL {_log_level.upper()} :: {_in_fstring_msg}")
    if _log_level == "debug":
        logging.debug(f"{_in_fstring_msg}")
    elif _log_level == "warning":
        logging.warning(f"{_in_fstring_msg}")
    elif _log_level == "info":
        logging.info(f"{_in_fstring_msg}")
    elif _log_level == "error":
        logging.error(f"{_in_fstring_msg}")
    elif _log_level == "critical":
        logging.critical(f"{_in_fstring_msg}")
    else:
        print(f"\n\n\nFATAL ERROR - wrong parameters passed to print_and_log function\n\n\nExiting with RC=9000\n")
        exit(9000)
    return

def check_command_line_args(_args):
    """
    Following checks performed:
    1) Mic input switch must be single character Y or N in lower or upper case.
            Will be converted to lowercase.
    2) If running with mic input:
        a) the directory to save the newly created wav files MUST exist.
        b) the directory to save the newly created file containing wav file locations MUST exist.
    3) If NOT running with mic input:
        a) the file containing wav file locations MUST exist.
    4) The directory to save the POS tagging info MUST exist.
    5) The directory to save the results of Image Captions after GUI (data for Story Generator to process) MUST exist.
    """
    ## extract command line args and check them
    micYorN           = args.mic_or_saved_wavs           ## -micYorN               parameter
    save_wavs_here    = args.save_wavs_here              ## -sw_here               parameter
    wavlocfile        = args.file_wav_loc_for_stt        ## -file_wav_loc_for_stt  parameter
    opfileallposinfo  = args.opfileallwordsposinfo       ## -op_4_stygen           parameter
    file4stygen       = args.opfile_for_sty_gen          ## -opfile_for_sty_gen    parameter
    logfileloc        = args.oplogfilelocation           ## -oplogfilelocation     parameter

    if (len(micYorN) != 1) or (micYorN not in ['Y', 'y', 'N', 'n']):
        return 10, f'Error in CLA parameter "-micYorN" : Expected upper/lowercase Y/N value', None
    micYorN = micYorN.lower()
    
    if save_wavs_here[-1] != r'/':
        save_wavs_here += r'/'
    if not os.path.isdir(save_wavs_here):
        return 15, f'Error in CLA parameter "-sw_here" : Expected an existing folder as input', None
    
    if not os.path.isdir( r'/'.join(wavlocfile.split(r'/')[:-1]) + r'/' ):
        return 20, f'Error in CLA parameter "-file4STT" : Expected an existing folder in the path specified before the filename', None
    
    if not os.path.isdir( r'/'.join(opfileallposinfo.split(r'/')[:-1]) + r'/' ):
        return 25, f'Error in CLA parameter "-opfileallposinfo" : Expected an existing folder in the path specified before the filename', None
    
    if not os.path.isdir( r'/'.join(file4stygen.split(r'/')[:-1]) + r'/' ):
        return 27, f'Error in CLA parameter "-file4stygen" : Expected an existing folder in the path specified before the filename', None
    
    if micYorN == 'n':
        ## check input file already exists
        if not os.path.isfile(wavlocfile):
            return 30, f'Error in CLA : If you do not want mic input then the "-file4STT" parameter MUST be an existing file', None
    
    return 0, None, [micYorN, save_wavs_here, wavlocfile, opfileallposinfo, file4stygen, logfileloc]

def execute_control_logic(args):
    
    DEBUG_SWITCH = False
    
    NEO4J_QUERY_RESULT_LIMIT = 20
    CANDIDATE_IMAGES_SELECTION_LIMIT = 5
    IMGCAP_INFERENCE_BEAM_WIDTH = 3

    check_cla_rc, check_cla_msg, returned_check_cla_returned_list = check_command_line_args(args)
    if check_cla_rc != 0:
        print(f"\n\nFATAL ERROR: Failed processing command line arguments with RC = {check_cla_rc}.\nMessage = {check_cla_msg}\n")
        print(f"\nExiting with Return Code = 10\n")
        exit(10)

    ## show CLA being used
    print(f"\nRunning with command line args as follows:")    
    cla_names = ('-micYorN', '-sw_here', '-file4STT', '-opfileallposinfo', '-file4stygen', '-oplogfilelocation')
    for idx, (cla_name, cla_value) in enumerate(zip(cla_names, returned_check_cla_returned_list)):
        print(f"""{idx+1}) {cla_name} = {''.join([ '"', cla_value, '"' ])}""")
    
    micYorN, save_wav_here, wavlocfile, opfileallposinfo, file4stygen, logfileloc = returned_check_cla_returned_list
    del check_cla_rc, check_cla_msg, returned_check_cla_returned_list

    ## setup logging file -   levels are DEBUG , INFO , WARNING , ERROR , CRITICAL
    logging.basicConfig(level=logging.DEBUG, filename=logfileloc,                               \
        filemode='w', format='LOG_LEVEL %(levelname)s : %(asctime)s :: %(message)s')
    
    #######################################################################
    ## STAGE 1 :: MIC INPUT + STT INFERENCE LOGIC
    #######################################################################
    myStr = "\n".join([
        f"\n\n-------------------------------------------------------------------",
        f"-------------------------------------------------------------------",
        f"    STAGE 1)  STARTING EXECUTION OF MIC INPUT + STT LOGIC          ",
        f"-------------------------------------------------------------------",
        f"-------------------------------------------------------------------\n\n"
        ])
    print_and_log(myStr, "info")
    myStr = None
    ## array for the STT logic results - to hold the filewise transcriptions
    mic_stt_logic_RC, mic_stt_logic_msg, mic_stt_module_results = mic_and_stt_functionality(micYorN, save_wav_here, wavlocfile, DEBUG_SWITCH)
    myStr = "\n".join([
        f"\nAfter MIC INPUT + STT logic execution:",
        f"mic_stt_logic_RC = {mic_stt_logic_RC}",
        f"mic_stt_logic_msg = {mic_stt_logic_msg}",
        f"mic_stt_module_results = {mic_stt_module_results}\n"
        ])
    print_and_log(myStr, "info")
    myStr = None

    #######################################################################
    ## STAGE 2 :: ID KEY ELEMENTS LOGIC
    #######################################################################
    if mic_stt_logic_RC == 0:
        myStr = "\n".join([
            f"\n\n-------------------------------------------------------------------",
            f"-------------------------------------------------------------------",
            f"    STAGE 2)  STARTING EXECUTION OF IDENTIFY KEYWORDS LOGIC        ",
            f"-------------------------------------------------------------------",
            f"-------------------------------------------------------------------\n\n"
            ])
        print_and_log(myStr, "info")
        myStr = None
        id_elements_logic_RC, id_elements_logic_msg, id_elements_module_results = id_elements_functionality(mic_stt_module_results, opfileallposinfo, DEBUG_SWITCH)
        myStr = "\n".join([
            f"\nAfter ID KEYWORDS AND SELECTION logic execution:",
            f"id_elements_logic_RC = {id_elements_logic_RC}",
            f"id_elements_logic_msg = {id_elements_logic_msg}",
            f"id_elements_module_results = {id_elements_module_results}\n"
            ])
        print_and_log(myStr, "info")
        myStr = None
    else:
        myStr = "\n".join([
            f"\n\nFATAL ERROR: MIC INPUT + STT LOGIC logic failed. Return code = {mic_stt_logic_RC}",
            f"\tMessage = {mic_stt_logic_msg}",
            f"\n\nPremature Exit with Exit Code = 50.\n\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(50)
    
    #######################################################################
    ## STAGE 3 :: QUERY NEO4J FOR IMAGES LOGIC
    #######################################################################
    if id_elements_logic_RC == 0:

        ## first check if user deselected all the candidate words for all the sentences.
        ##         in such scenario  there is nothing more to process!!!
        if all(ele== [] for ele in id_elements_module_results):
            myStr = "\n".join([
                f"\n\nFATAL ERROR: User deselected all words of all sentences during candidate keywords selection stage.",
                f"No downstream processing possible for Query Database stage.\n\n"
                ])
            print_and_log(myStr, 'error')
            return
        
        ## user did not deselect all the words - so continue dowstream processing
        myStr = "\n".join([
            f"\n\n-------------------------------------------------------------------",
            f"-------------------------------------------------------------------",
            f"    STAGE 3)  STARTING EXECUTION OF QUERY NEO4J LOGIC              ",
            f"-------------------------------------------------------------------",
            f"-------------------------------------------------------------------\n\n"
            ])
        print_and_log(myStr, "info")
        myStr = None
        query_neo4j_logic_RC, query_neo4j_logic_msg, query_neo4j_module_results = query_neo4j_functionality(id_elements_module_results, NEO4J_QUERY_RESULT_LIMIT, DEBUG_SWITCH)
    else:
        myStr = "\n".join([
            f"\n\nFATAL ERROR: Identify keywords logic failed. Return code = {query_neo4j_logic_RC}",
            f"\tMessage = {query_neo4j_logic_msg}",
            f"\nPremature Exit with Exit Code = 200\n\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(200)

    ## show the result if no problem - for readability, match the keywords against the candidate images retrieved from neo4j query
    if query_neo4j_logic_RC == 0:
        myStr= f"\nAfter QUERY NEO4J logic execution, Database query results - Candidate Images at this stage:\n"
        print_and_log(myStr, "info")
        myStr = None
        for idx, (v1, v2) in enumerate(zip(id_elements_module_results, query_neo4j_module_results)):
            myStr= f"\nQuery {idx+1}) Keywords: {v1}\nQuery result:\n{v2}"
            print_and_log(myStr, "info")
    else:
        myStr = "\n".join([
            f"\n\nFATAL ERROR: Database query logic failed. Return code = {query_neo4j_logic_RC}",
            f"\tMessage = {query_neo4j_logic_msg}",
            f"\nPremature Exit with Exit Code = 300\n\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(300)
    
    #######################################################################
    ## STAGE 4-a :: GUI FOR IMAGE SELECTION JUST AFTER NEO4J QUERY LOGIC
    #######################################################################
    if query_neo4j_logic_RC == 0:
        myStr = "\n".join([
            f"\n\n-------------------------------------------------------------------",
            f"-------------------------------------------------------------------",
            f"    STAGE 4)  STARTING EXECUTION OF IMAGE SELECTION VIA GUI        ",
            f"-------------------------------------------------------------------",
            f"-------------------------------------------------------------------\n\n"
            ])
        print_and_log(myStr, "info")
        myStr = None
        gui_candidate_image_selection_logic_RC, gui_candidate_image_selection_logic_msg, gui_candidate_image_selection_module_results = gui_candidate_image_selection_functionality(query_neo4j_module_results, DEBUG_SWITCH)
        myStr = "\n".join([
            f"\nAfter GUI CANDIDATE IMAGES SELECTION logic execution:",
            f"gui_candidate_image_selection_logic_RC = {gui_candidate_image_selection_logic_RC}",
            f"gui_candidate_image_selection_logic_msg = {gui_candidate_image_selection_logic_msg}",
            f"gui_candidate_image_selection_module_results = {gui_candidate_image_selection_module_results}\n"
            ])
        print_and_log(myStr, "info")
        myStr = None
    else:                       ## actually this will never happen coz the check for query neo4j logic RC == 0 already done earlier and will exit if not 0. Keeping to maintain readability of code
        myStr = "\n".join([
            f"\n\nFATAL ERROR: Database query logic failed. Return code = {query_neo4j_logic_RC}",
            f"\tMessage = {query_neo4j_logic_msg}",
            f"\nPremature Exit with Exit Code = 300\n\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(300)
    
    ## show the result if no problem - for readability, match the keywords against the images finally selected by user after gui presentation
    myStr = f"\nImages retained after Deselections (to be passed to Auto-caption block):\n"
    print_and_log(myStr, "info")
    myStr = None
    if gui_candidate_image_selection_logic_RC == 0:
        for idx, (v1, v2) in enumerate(zip(id_elements_module_results, gui_candidate_image_selection_module_results)):
            myStr = "\n".join([
                f"\n{idx+1}) Keywords: {v1}",
                f"Selected Images results:\n{v2}"
                ])
            print_and_log(myStr, "info")
            myStr = None
    else:
        myStr = "\n".join([
            f"\n\nFATAL ERROR: GUI based candidate image Deselections failed. Return code = {gui_candidate_image_selection_logic_RC}",
            f"\tMessage = {gui_candidate_image_selection_logic_msg}",
            f"\nPremature Exit with Exit Code = 400\n\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(400)
    
    #######################################################################
    ## STAGE 4-b :: MAKE SURE ALL IMAGES FOR ALL QUERIES HAVE NOT BEEN DESLECTED
    #######################################################################
    ## First check if user deselected all the candidate images for all the sentences
    ##               in such scenario  there is nothing more to process!!!
    ## Next, if there are more than 5 images selected for any of the sentences, then stop processing further
    ##               as that should have been impossible to occur.
    if all( ele== [] for ele in gui_candidate_image_selection_module_results ):
        myStr = "\n".join([
            f"\n\nFATAL ERROR: User Deselected all images of all queries during candidate images selection stage.",
            f"No downstream processing possible for Auto Caption stage.\n\n"
            ])
        print_and_log(myStr, 'error')
        return
    elif any( len(ele) > CANDIDATE_IMAGES_SELECTION_LIMIT for ele in gui_candidate_image_selection_module_results ):
        ## this can happen if user closes root window without actually doing the image selection. Whatever number of candidate images there were will be retained
        ##      as is and count can be greater than allowed limit.
        ## make sure    self.max_limit_images_selections   in    c_queryImgSelection_grid_wnd_window   has same value as    CANDIDATE_IMAGES_SELECTION_LIMIT 
        myStr = "\n".join([
            f"\n\nFATAL ERROR: User Selected more than {CANDIDATE_IMAGES_SELECTION_LIMIT} candidate images during selection stage.",
            f"Cannot proceed to Image Captioning stage.\n\n",
            f"\nPremature Exit with Exit Code = 420\n\n"
            ])
        print_and_log(myStr, 'error')
        myStr = None
        exit(420)
    
    #######################################################################
    ## STAGE 5-a :: IMAGE CAPTIONING - NO ATTENTION - INFERENCE LOGIC
    #######################################################################
    ## do the auto caption No-attention logic processing on the images selected in previous stage - errors checked already
    myStr = "\n".join([
        f"\n\n-------------------------------------------------------------------",
        f"-------------------------------------------------------------------",
        f"   STAGE 5-a)  STARTING EXECUTION OF IMAGE CAPTIONING - NO ATTN    ",
        f"-------------------------------------------------------------------",
        f"-------------------------------------------------------------------\n\n"
        ])
    print_and_log(myStr, 'info')
    myStr = None
    img_cap_after_no_attn_inference_logic_RC, img_cap_after_no_attn_inference_logic_msg, img_cap_after_no_attn_inference_module_results = img_cap_no_attn_inference_functionality(id_elements_module_results, gui_candidate_image_selection_module_results, IMGCAP_INFERENCE_BEAM_WIDTH, DEBUG_SWITCH)
    myStr = "\n".join([
        f"\nAfter IMAGE CAPTIONING NO-ATTENTION logic execution:",
        f"img_cap_after_no_attn_inference_logic_RC = {img_cap_after_no_attn_inference_logic_RC}",
        f"img_cap_after_no_attn_inference_logic_msg = {img_cap_after_no_attn_inference_logic_msg}",
        f"img_cap_after_no_attn_inference_module_results = {img_cap_after_no_attn_inference_module_results}\n"
        ])
    print_and_log(myStr, "info")
    myStr = None
    
    #######################################################################
    ## STAGE 5-b :: IMAGE CAPTIONING - WITH ATTENTION - INFERENCE LOGIC
    #######################################################################
    ## do the auto caption WITH-attention logic processing on the images selected - only if no attention logic successful
    if img_cap_after_no_attn_inference_logic_RC == 0:
        myStr = "\n".join([
            f"\n\n-------------------------------------------------------------------",
            f"-------------------------------------------------------------------",
            f"  STAGE 5-b)  STARTING EXECUTION OF IMAGE CAPTIONING - WITH ATTN   ",
            f"-------------------------------------------------------------------",
            f"-------------------------------------------------------------------\n\n"
            ])
        print_and_log(myStr, 'info')
        myStr = None
        img_cap_after_WITH_attn_inference_logic_RC, img_cap_after_WITH_attn_inference_logic_msg, img_cap_after_WITH_attn_inference_module_results = img_cap_WITH_attn_inference_functionality(img_cap_after_no_attn_inference_module_results, IMGCAP_INFERENCE_BEAM_WIDTH, DEBUG_SWITCH)
        myStr = "\n".join([
            f"\nAfter IMAGE CAPTIONING WITH-ATTENTION logic execution:",
            f"img_cap_after_WITH_attn_inference_logic_RC = {img_cap_after_WITH_attn_inference_logic_RC}",
            f"img_cap_after_WITH_attn_inference_logic_msg = {img_cap_after_WITH_attn_inference_logic_msg}",
            f"img_cap_after_WITH_attn_inference_module_results = {img_cap_after_WITH_attn_inference_module_results}\n"
            ])
        print_and_log(myStr, "info")
        myStr = None
    else:
        myStr = "\n".join([
            f"\n\nFATAL ERROR: Image captioning No-Attention logic failed. Return code = {img_cap_after_no_attn_inference_logic_RC}",
            f"\tMessage = {img_cap_after_no_attn_inference_logic_msg}",
            f"\nPremature Exit with Exit Code = 450\n\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(450)

    #######################################################################
    ## STAGE 6 :: GUI FOR EDITING CAPTIONS AND/OR DESELECTION OF IMAGES LOGIC - BOTH MODELS
    #######################################################################
    ## gui display the images and their captions - for NO-ATTENTION logic output
    if img_cap_after_WITH_attn_inference_logic_RC == 0:
        myStr = "\n".join([
            f"\n\n-------------------------------------------------------------------------------",
            f"-------------------------------------------------------------------------------",
            f"    STAGE 6)  STARTING EXECUTION OF GUI DISPLAY fOR IMAGE CAPTIONING RESULTS   ",
            f"-------------------------------------------------------------------------------",
            f"-------------------------------------------------------------------------------\n\n"
            ])
        print_and_log(myStr, "info")
        myStr = None
        gui_img_cap_show_results_logic_RC, gui_img_cap_show_results_logic_msg, gui_img_cap_module_results = gui_img_cap_show_gui_functionality(img_cap_after_WITH_attn_inference_module_results, file4stygen, DEBUG_SWITCH)
        myStr = "\n".join([
            f"\nAfter GUI TO DISPLAY IMAGE CAPTIONING RESULTS logic execution:",
            f"gui_img_cap_show_results_logic_RC = {gui_img_cap_show_results_logic_RC}",
            f"gui_img_cap_show_results_logic_msg =\n{gui_img_cap_show_results_logic_msg}",
            f"gui_img_cap_module_results =\n{gui_img_cap_module_results}\n"
            ])
        print_and_log(myStr, "info")
        myStr = None
    else:
        myStr = "\n".join([
            f"\n\nFATAL ERROR: Image captioning WITH-Attention logic failed. Return code = {img_cap_after_WITH_attn_inference_logic_RC}",
            f"\tMessage = {img_cap_after_WITH_attn_inference_logic_msg}",
            f"\nPremature Exit with Exit Code = 500.\n\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(500)
    
    if gui_img_cap_show_results_logic_RC != 0:
        myStr = "\n".join([
            f"\n\nFATAL ERROR: GUI Image Captioning logic failed. Return code = {gui_img_cap_show_results_logic_RC}",
            f"\tMessage = {gui_img_cap_show_results_logic_msg}",
            f"\nPremature Exit with Exit Code = 550.\n\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(550)
    
    return

if __name__ == '__main__':
    args = argparser.parse_args()
    execute_control_logic(args)

    myStr = f"\n\n\nNormal exit from program.\n"
    print_and_log(myStr, "info")
    myStr = None

    exit(0)
