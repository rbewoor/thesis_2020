## -----------------------------------------------------------------------------
###             FUNCTIONAL programming approach
### Goal: Process the output of the edited or not Image Caption stage output.
###       Use the captions to generate the short story using the Story Generator.
## -----------------------------------------------------------------------------
##  STAGE 7) Story Generation using fine-tuned GPT-2 model (355M).
##        Use the captions (original or edited as the case may be) to generate short story.
##        Logic:
##           a) Make combinations of seed sentences by taking on sentence for each query.
##           b) Use the combination seed sentences to create on story.
##           c) Create multiple stories by using different combinations of seed sentences.
##           d) Score the stories for grammar, etc.
##       Input:
##        a) Intermediate data from previous stage:
##           File containing the data structure as input.
##       Outputs:
##        a) Text file containing the generated texts.
##
## -----------------------------------------------------------------------------
## Command line arguments:
## 1) -ipf4stygen      : Input file containing results of previous stage after image captioning and gui presentation.
## 2) -ckptdir          : Checkpoint directory from which to restore previous trained GPT-2 model for inference.
## 3) -run              : Which run tag files to be loaded for checkpoints.
##                            Has both captions generator outputs.
## 4) -op_res_file      : File to save the results of generated stories.
## 5) -len              : Length of output required
## 6) -temp             : Temperature value
## 7) -nsamp            : Number of samples
## 8) -batsz            : Batch size
## 9) -topk             : Interger value controlling diversity
## 10)-logfileloc       : Location to write the log file.
## -----------------------------------------------------------------------------
## Usage example:
## 
## python3 StyGen_Inference_4A.py -ipf4stygen "/home/rohit/PyWDUbuntu/thesis/combined_execution/ImgCapAfterGui/op_img_cap_after_gui.txt" -ckptdir "/home/rohit/PyWDUbuntu/thesis/StyGen/checkpoint/" -run "Run2_File11_2_checkpoint_run2" -op_res_file "/home/rohit/PyWDUbuntu/thesis/combined_execution/StyGen/op_sty_gen_1.txt" -logfileloc "./LOG_StyGen_Inference_4A.LOG" -nsamp "1" -batsz "1" -len "300" -temp "0.95" -topk "0"
## -----------------------------------------------------------------------------

## import necessary packages

##   imports for common or generic use packages
#from __future__ import print_function
import argparse
import os
import json
import time
import datetime
import subprocess
#import numpy as np
import copy
import logging

##   tensorflow related
import tensorflow as tf
from tensorflow import keras

##   imports for story generation
import gpt_2_simple as gpt2
from gpt_2_simple.src import model, sample, encoder, memory_saving_gradients
from gpt_2_simple.src.load_dataset import load_dataset, Sampler
from gpt_2_simple.src.accumulate import AccumulatingOptimizer

## command line arguments
argparser = argparse.ArgumentParser(
    description='parameters to run this program')

argparser.add_argument(
    '-ipf4stygen',
    '--ipfile_for_sty_gen',
    help='location to pick results for story generator logic to process')

argparser.add_argument(
    '-op_res_file',
    '--opfile_for_sty_gen_results',
    help='location to store the data structure of images and their stories')

argparser.add_argument(
    '-ckptdir',
    '--checkpoint_directory',
    help='location of GPT-2 checkpoint directory')

argparser.add_argument(
    '-run',
    '--run_name',
    help='folder name of run in the checkpoint directory')

argparser.add_argument(
    '-len',
    '--length',
    type=int,
    choices=range(100,501),
    help='interger value from 100 to 500 specifying how long the output should be')

argparser.add_argument(
    '-temp',
    '--temperature',
    help='float from 0.0 to 1.0 with zero meaning totally deterministic and 1.0 being max variability')

argparser.add_argument(
    '-nsamp',
    '--number_of_samples',
    type=int,
    choices=range(1, 6),
    help='integer from 1 to 5 specifying number of stories required per input seed. Note that nsamp must be divisible by batsz')

argparser.add_argument(
    '-batsz',
    '--batch_size',
    type=int,
    help='integer for batch size. Note that nsamp must be divisible by batsz')

argparser.add_argument(
    '-topk',
    '--top_k',
    type=int,
    help='integer controlling diversity. 0 means no restriction, 1 means only one word considered, typically value of 40 used.')

argparser.add_argument(
    '-logfileloc',
    '--oplogfilelocation',
    help='location for output file for logging')

## ----------------------------------------------------------------------------------------------------------------------

###############################################################################################################
## -------------------     STORY GENERATOR STARTS                  STORY GENERATOR STARTS   -------------------
###############################################################################################################
def my_load_gpt2(_sess,
              _checkpoint='latest',
              _run_name="run1",
              _checkpoint_dir="checkpoint",
              _model_name=None,
              _model_dir='models',
              _multi_gpu=False):
    """
    Loads the model checkpoint or existing model into a TensorFlow session for repeated predictions.
    """

    if _model_name:
        checkpoint_path = os.path.join(_model_dir, _model_name)
    else:
        checkpoint_path = os.path.join(_checkpoint_dir, _run_name)
    print(f"\ncheckpoint_path = {checkpoint_path}\n")

    hparams = model.default_hparams()
    with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    context = tf.compat.v1.placeholder(tf.int32, [1, None])

    gpus = []
    if _multi_gpu:
        gpus = get_available_gpus()

    output = model.model(hparams=hparams, X=context, gpus=gpus)

    if _checkpoint=='latest':
        ckpt = tf.train.latest_checkpoint(checkpoint_path)
    else:
        ckpt = os.path.join(checkpoint_path,_checkpoint)

    saver = tf.compat.v1.train.Saver(allow_empty=True)
    _sess.run(tf.compat.v1.global_variables_initializer())

    if _model_name:
        print(f"\nLoading pretrained model :: {ckpt}\n")
    else:
        print(f"\nLoading checkpoint :: {ckpt}\n")
    saver.restore(_sess, ckpt)

def reset_session(_sess, threads=-1, server=None):
    """Resets the current TensorFlow session, to clear memory
    or load another model.
    """

    tf.compat.v1.reset_default_graph()
    _sess.close()

def stg_gen_do_one_inference(_seed_string, _ckptdir, _runName, _sg_len, _sg_temp, _sg_nsamp, _sg_batsz, _topk, _DEBUG_SWITCH):
    ## Parameters to generate the story
    ## NOTE: nsamples % batch_size == 0
    # SG_LENGTH = 300
    # SG_TEMPERATURE = 0.95
    # SG_NSAMPLES = 1
    # SG_BATCH_SIZE = 1
    SG_LENGTH = _sg_len
    SG_TEMPERATURE = _sg_temp
    SG_NSAMPLES = _sg_nsamp
    SG_BATCH_SIZE = _sg_batsz
    SG_TOP_K = _topk
    print(f"\nlength = {SG_LENGTH}\ttemp = {SG_TEMPERATURE}\tnsamp = {SG_NSAMPLES}\tbtch_siz = {SG_BATCH_SIZE}\ttopk = {SG_TOP_K}\n")
    # RETURN_LIST = True ## hardcoded as True during function call

    ## RUN_NAME must match a folder in the checkpoint directory
    #RUN_NAME = r"Run2_File11_2_checkpoint_run2"

    ## check samples and batch size values are compatible
    assert (SG_NSAMPLES % SG_BATCH_SIZE == 0) , f"Values for NSAMPLES and BATCH_SIZE incompatible. " \
            f"NSAMPLES={SG_NSAMPLES} % SG_BATCH_SIZE={SG_BATCH_SIZE} != 0"
    
    ## set up the pre-trained GPT-2 model from checkpoint directory
    
    ## copy checkpoint directory from gdrive - not required as saved it locally
    ##gpt2.copy_checkpoint_from_gdrive(run_name='run1')

    sess = gpt2.start_tf_sess()
    myStr = f"\nTF session started\n"
    print_and_log(myStr, "debug")
    myStr = None
    
    my_load_gpt2(
        _sess = sess,
        _checkpoint='latest',
        _run_name=_runName,
        _checkpoint_dir=_ckptdir,
        _model_name=None,
        _model_dir='models',
        _multi_gpu=False)
    
    myStr = f"\n\n\tGenerating the story using 'prefix' =\n{_seed_string}\n"
    print_and_log(myStr, "info")
    myStr = None

    story_as_list = gpt2.generate(
        sess,
        length=SG_LENGTH,
        temperature=SG_TEMPERATURE,
        top_k=SG_TOP_K,
        prefix=_seed_string,
        nsamples=SG_NSAMPLES,
        batch_size=SG_BATCH_SIZE,
        checkpoint_dir=_ckptdir,
        return_as_list=True
        )
    
    reset_session(sess)

    ## for each story, split on new line and rejoin with space
    for i, each_story in enumerate(story_as_list):
        split_sents = each_story.split('\n')
        story_as_list[i] = ' '.join(split_sents)

    myStr = f"\n\n\tGENERATED STORY after removing newline characters =\n{story_as_list}\n\n"
    print_and_log(myStr, "info")
    myStr = None
    return story_as_list

def sty_gen_create_triplets(_prev_stage_info, _attn_caps_switch, _DEBUG_SWITCH):
    if _attn_caps_switch == "N":
        #list1, list2, list3 = [ list(range( len(q_info['selected_images_no_attn']) )) for q_info in _prev_stage_info]
        try:
            list1 = list(range(len(_prev_stage_info[0]['selected_images_no_attn'])))
        except IndexError:
            list1 = list()
        try:
            list2 = list(range(len(_prev_stage_info[1]['selected_images_no_attn'])))
        except IndexError:
            list2 = list()
        try:
            list3 = list(range(len(_prev_stage_info[2]['selected_images_no_attn'])))
        except IndexError:
            list3 = list()
    elif _attn_caps_switch == "Y":
        #list1, list2, list3 = [ list(range( len(q_info['selected_images_with_attn']) )) for q_info in _prev_stage_info]
        try:
            list1 = list(range(len(_prev_stage_info[0]['selected_images_with_attn'])))
        except IndexError:
            list1 = list()
        try:
            list2 = list(range(len(_prev_stage_info[1]['selected_images_with_attn'])))
        except IndexError:
            list2 = list()
        try:
            list3 = list(range(len(_prev_stage_info[2]['selected_images_with_attn'])))
        except IndexError:
            list3 = list()
    myStr = '\n'.join([
        f"\nLists for triplets BEFORE inserting -1:",
        f"list1 = \n{list1}",
        f"list2 = \n{list2}",
        f"list3 = \n{list3}\n"
        ])
    print_and_log(myStr, "info")
    myStr = None
    print(f"list1 = {list1}\nlist2 = {list2}\nlist3 = {list3}\n")
    ## insert -1 as dummy value into empty lists. Prevents the for loop later from not executing
    if not list1: list1.append(-1)
    if not list2: list2.append(-1)
    if not list3: list3.append(-1)
    myStr = '\n'.join([
        f"\nLists for triplets AFTER inserting -1:",
        f"list1 = \n{list1}",
        f"list2 = \n{list2}",
        f"list3 = \n{list3}\n"
        ])
    print_and_log(myStr, "info")
    myStr = None
    results = list()
    for v1 in list1:
        for v2 in list2:
            for v3 in list3:
                results.append((v1, v2, v3))
    return results

def sty_gen_inference_functionality(_ipf4stygen, _ckptdir, _runName, _opfile, _sg_len, _sg_temp, _sg_nsamp, _sg_batsz, _topk, _DEBUG_SWITCH):
    ## load the data of previous stage from the saved file
    try:
        with  open(_ipf4stygen, "r") as f:
            gui_img_cap_module_results = json.load(f)
        myStr = "\n".join([
            f"\nSuccessfully reloaded data from previous stage",
            f"gui_img_cap_module_results =\n{gui_img_cap_module_results}\n"
            ])
        print_and_log(myStr, "info")
        myStr = None
    except Exception as stg_gen_ipfile_reload_problem_msg:
        myStr = '\n'.join([
            f"\nFATAL ERROR: Problem reloading data from previous stage from file =\n{_file4stygen}",
            f"Error message =\n{stg_gen_ipfile_reload_problem_msg}",
            f"Exiting with Return Code = 200\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(200)
    
    """
    ##### temp code - start - no empty list
    temp_no_empty_gui_img_cap_module_results = \
    [
        {
            "key_elements": ["person", "bicycle"],
            "selected_images_no_attn":
            [
                ["/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000155758.jpg", "man riding bike on the side of the road with child in carriage behing it"],
                ["/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000056139.jpg", "man riding bike next to man on beach"]
                ],
            "selected_images_with_attn":
            [
                ["/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000155758.jpg", "a man and woman are riding a bike with a dog"],
                ["/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000056139.jpg", "a woman with a bicycle on a beach with some kites flying behind her"]
            ]
            },
        {
            "key_elements": ["person", "tvmonitor"],
            "selected_images_no_attn":
            [
                ["/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000382917.jpg", "QUERY 2 - IMAGE 1 - EDITED STANDARD CAPTION"],
                ["/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000438521.jpg", "man and woman are playing video game"]
            ],
            "selected_images_with_attn":
            [
                ["/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000382917.jpg", "QUERY 2 - IMAGE 1 - EDITED ATTENTION CAPTION"],
                ["/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000438521.jpg", "QUERY 2 - IMAGE 2 - EDITED ONLY ATTENTION CAPTION"]
                ]
            },
        {
            "key_elements": ["handbag"],
            "selected_images_no_attn": [],
            "selected_images_with_attn": []
            }
        ]
    #gui_img_cap_module_results = temp_no_empty_gui_img_cap_module_results
    ##### temp code - end
    ##### temp code - start - with empty list
    temp_empty_list_start_gui_img_cap_module_results = \
    [
        {
            "key_elements": ["handbag"],
            "selected_images_no_attn": [],
            "selected_images_with_attn": []
            },
        {
            "key_elements": ["person", "bicycle"],
            "selected_images_no_attn":
            [
                ["/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000155758.jpg", "man riding bike on the side of the road with child in carriage behing it"],
                ["/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000056139.jpg", "man riding bike next to man on beach"]
                ],
            "selected_images_with_attn":
            [
                ["/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000155758.jpg", "a man and woman are riding a bike with a dog"],
                ["/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000056139.jpg", "a woman with a bicycle on a beach with some kites flying behind her"]
            ]
            },
        {
            "key_elements": ["person", "tvmonitor"],
            "selected_images_no_attn":
            [
                ["/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000382917.jpg", "QUERY 2 - IMAGE 1 - EDITED STANDARD CAPTION"],
                ["/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000438521.jpg", "man and woman are playing video game"]
            ],
            "selected_images_with_attn":
            [
                ["/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000382917.jpg", "QUERY 2 - IMAGE 1 - EDITED ATTENTION CAPTION"],
                ["/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000438521.jpg", "QUERY 2 - IMAGE 2 - EDITED ONLY ATTENTION CAPTION"]
                ]
            }
        ]
    #gui_img_cap_module_results = temp_empty_list_start_gui_img_cap_module_results
    ##### temp code - end
    """

    ## make copy for results of this stage - add empty list to store generated stories
    sty_gen_inference_module_results = copy.deepcopy(gui_img_cap_module_results)

    
    ## create sets of all combinations possible containing exactly one image from each of the queries
    ## Note: -1 is dummy to indicate no image for that query
    
    ## first for NO ATTENTION CAPTIONS - set swtich to N
    attn_caps_switch = "N"
    triplets_no_attn = sty_gen_create_triplets(gui_img_cap_module_results, attn_caps_switch, _DEBUG_SWITCH)
    myStr = f"\nTriplets_NO_ATTN BEFORE any -1 insertions =\n{triplets_no_attn}\n"
    print_and_log(myStr, "error")
    myStr = None

    triplet_no_attn_seeds_for_styGen = []
    triplet_no_attn_imgs_for_stgGen = []
    for triplet in triplets_no_attn:
        trip_no_attn_seeds = []
        trip_no_attn_imgs  = []
        q1idx, q2idx, q3idx = triplet
        if q1idx == -1:
            trip_no_attn_seeds.append(None)
            trip_no_attn_imgs.append(None)
        else:
            trip_no_attn_seeds.append(gui_img_cap_module_results[0]['selected_images_no_attn'][q1idx][1].lower().capitalize() + r'.' )# + r' ')
            trip_no_attn_imgs.append(gui_img_cap_module_results[0]['selected_images_no_attn'][q1idx][0])
        if q2idx == -1:
            trip_no_attn_seeds.append(None)
            trip_no_attn_imgs.append(None)
        else:
            trip_no_attn_seeds.append(gui_img_cap_module_results[1]['selected_images_no_attn'][q2idx][1].lower().capitalize() + r'.' )# + r' ')
            trip_no_attn_imgs.append(gui_img_cap_module_results[1]['selected_images_no_attn'][q2idx][0])
        if q3idx == -1:
            trip_no_attn_seeds.append(None)
            trip_no_attn_imgs.append(None)
        else:
            trip_no_attn_seeds.append(gui_img_cap_module_results[2]['selected_images_no_attn'][q3idx][1].lower().capitalize() + r'.' )# + r' ')
            trip_no_attn_imgs.append(gui_img_cap_module_results[2]['selected_images_no_attn'][q3idx][0])
        triplet_no_attn_seeds_for_styGen.append(trip_no_attn_seeds)
        triplet_no_attn_imgs_for_stgGen.append(trip_no_attn_imgs)
    
    myStr = f"\nTriplets_NO_ATTN AFTER any -1 insertions =\n{triplet_no_attn_seeds_for_styGen}\nAll imgs =\n{triplet_no_attn_imgs_for_stgGen}\nTotal NO ATTN triples = {len(triplet_no_attn_seeds_for_styGen)}\n"
    print_and_log(myStr, "info")
    myStr = None

    ## next for WITH ATTENTION CAPTIONS - set swtich to Y
    attn_caps_switch = "Y"
    triplets_with_attn = sty_gen_create_triplets(gui_img_cap_module_results, attn_caps_switch, _DEBUG_SWITCH)
    myStr = f"\nTriplets_WITH_ATTN BEFORE any -1 insertions =\n{triplets_with_attn}\n"
    print_and_log(myStr, "error")
    myStr = None

    triplet_with_attn_seeds_for_styGen = []
    triplet_with_attn_imgs_for_stgGen = []
    for triplet in triplets_with_attn:
        trip_with_attn_seeds = []
        trip_with_attn_imgs  = []
        q1idx, q2idx, q3idx = triplet
        if q1idx == -1:
            trip_with_attn_seeds.append(None)
            trip_with_attn_imgs.append(None)
        else:
            trip_with_attn_seeds.append(gui_img_cap_module_results[0]['selected_images_with_attn'][q1idx][1].lower().capitalize() + r'.' )# + r' ')
            trip_with_attn_imgs.append(gui_img_cap_module_results[0]['selected_images_with_attn'][q1idx][0])
        if q2idx == -1:
            trip_with_attn_seeds.append(None)
            trip_with_attn_imgs.append(None)
        else:
            trip_with_attn_seeds.append(gui_img_cap_module_results[1]['selected_images_with_attn'][q2idx][1].lower().capitalize() + r'.' )# + r' ')
            trip_with_attn_imgs.append(gui_img_cap_module_results[1]['selected_images_with_attn'][q2idx][0])
        if q3idx == -1:
            trip_with_attn_seeds.append(None)
            trip_with_attn_imgs.append(None)
        else:
            trip_with_attn_seeds.append(gui_img_cap_module_results[2]['selected_images_with_attn'][q3idx][1].lower().capitalize() + r'.' )# + r' ')
            trip_with_attn_imgs.append(gui_img_cap_module_results[2]['selected_images_with_attn'][q3idx][0])
        triplet_with_attn_seeds_for_styGen.append(trip_with_attn_seeds)
        triplet_with_attn_imgs_for_stgGen.append(trip_with_attn_imgs)
    
    myStr = f"\nTriplets_WITH_ATTN AFTER any -1 insertions =\n{triplet_with_attn_seeds_for_styGen}\nAll imgs =\n{triplet_with_attn_imgs_for_stgGen}\nTotal WITH ATTN triples = {len(triplet_with_attn_seeds_for_styGen)}\n"
    print_and_log(myStr, "info")
    myStr = None
    
    ## make copy for results of this stage
    sty_gen_inference_module_results = copy.deepcopy(gui_img_cap_module_results)
    ## add dict entries of empty lists to store generated stories - later tuples of story and associated image/s will be added to this
    ##     make entry to hold the combined sentences stories for NO ATTN
    sty_gen_inference_module_results.append({'combined_no_attn_sent_stories': []}) ## fourth entry i.e. index=3 (1st, 2nd, 3rd are query info)
    ##     make entry to hold the combined sentences stories for WITH ATTN
    sty_gen_inference_module_results.append({'combined_with_attn_sent_stories': []}) ## fifth entry i.e. index=4
    
    ## actually generate the stories from model
    ## 1) combined sentences stories for NO ATTN
    myStr = f"\n\nStarted generation NO ATTENTION stories.\n\n"
    print_and_log(myStr, "info")
    myStr = None
    cnt_outer_loop = len(triplet_no_attn_seeds_for_styGen)
    for idx1, (trip_seeds, trip_imgs) in enumerate(zip(triplet_no_attn_seeds_for_styGen, triplet_no_attn_imgs_for_stgGen)):
        myStr = f"\n\nOUTER Loop {idx1 + 1} of {cnt_outer_loop}\n\n"
        print_and_log(myStr, "info")
        myStr = None
        comb_seed = ' '.join(seed for seed in trip_seeds if seed is not None)
        story_texts = stg_gen_do_one_inference(comb_seed, _ckptdir, _runName, _sg_len, _sg_temp, _sg_nsamp, _sg_batsz, _topk, _DEBUG_SWITCH)
        cnt_inner_loop = len(story_texts)
        for idx2, each_story in enumerate(story_texts):
            myStr = f"\n\nINNER Loop {idx2 + 1} of {cnt_inner_loop}\n\n"
            print_and_log(myStr, "info")
            myStr = None
            ## 2nd last position is for comb no attn
            sty_gen_inference_module_results[-2]['combined_no_attn_sent_stories'].append(([img for img in trip_imgs if img is not None], each_story))
    ## 2) combined sentences stories for WITH ATTN
    myStr = f"\n\nStarted generation WITH ATTENTION stories.\n\n"
    print_and_log(myStr, "info")
    myStr = None
    cnt_outer_loop = len(triplet_with_attn_seeds_for_styGen)
    for idx1, (trip_seeds, trip_imgs) in enumerate(zip(triplet_with_attn_seeds_for_styGen, triplet_with_attn_imgs_for_stgGen)):
        myStr = f"\n\nOUTER Loop {idx1 + 1} of {cnt_outer_loop}\n\n"
        print_and_log(myStr, "info")
        myStr = None
        comb_seed = ' '.join(seed for seed in trip_seeds if seed is not None)
        story_texts = stg_gen_do_one_inference(comb_seed, _ckptdir, _runName, _sg_len, _sg_temp, _sg_nsamp, _sg_batsz, _topk, _DEBUG_SWITCH)
        cnt_inner_loop = len(story_texts)
        for idx2, each_story in enumerate(story_texts):
            myStr = f"\n\nINNER Loop {idx2 + 1} of {cnt_inner_loop}\n\n"
            print_and_log(myStr, "info")
            myStr = None
            ## last position is for comb with attn
            sty_gen_inference_module_results[-1]['combined_with_attn_sent_stories'].append(([img for img in trip_imgs if img is not None], each_story))
    
    ## save results data structure to output file
    try:
        with open(_opfile, "w") as f:
            json.dump(sty_gen_inference_module_results, f)
        myStr = f"\nSaved results data structure to file here: {_opfile}\n"
        print_and_log(myStr, "info")
        myStr = None
    except Exception as opfile_save_results_problem:
        myStr = '\n'.join([
            f"\nFATAL ERROR: Problem saving results data structure to file here: {_opfile}",
            f"Error message =\n{opfile_save_results_problem}",
            f"Exiting with Return Code = 250\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(250)

    return (0, None, sty_gen_inference_module_results)

###############################################################################################################
## ---------------------     STORY GENERATOR ENDS                  STORY GENERATOR ENDS   ---------------------
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
    1) The input file for story generator  MUST exist.
    2) The directory for checkpoint folder MUST exist.
    3) The run name MUST be a directory in the checkpoint folder.
    4) The directory for the output results file MUST exist.
    """
    ## extract command line args and check them
    ipf4stygen        = args.ipfile_for_sty_gen          ## -file4stygen           parameter
    opfile            = args.opfile_for_sty_gen_results  ## -op_res_file           parameter
    ckptdir           = args.checkpoint_directory        ## -ckptdir               parameter
    runName           = args.run_name                    ## -run                   parameter
    sg_len            = args.length                      ## -len                   parameter
    sg_temp           = args.temperature                 ## -temp                  parameter
    sg_nsamp          = args.number_of_samples           ## -nsamp                 parameter
    sg_batsz          = args.batch_size                  ## -batsz                 parameter
    sg_topk           = args.top_k                       ## -topk                  parameter
    logfileloc        = args.oplogfilelocation           ## -oplogfilelocation     parameter

    if not os.path.isfile(ipf4stygen):
        return 10, f'Error in CLA parameter "-ipf4stygen" : Input file does not exist.', None
    
    if ckptdir[-1] != r'/':
        ckptdir += r'/'
    if not os.path.isdir(ckptdir):
        return 10, f'Error in CLA parameter "-ckptdir" : Checkpoint folder does not exist.', None
    
    runDir = ckptdir + runName
    if runDir[-1] != r'/':
        runDir += r'/'
    if not os.path.isdir(runDir):
        cla_err_msg = f'Error in CLA parameter/s "-ckptdir and/or -run" : The run paramter must be a folder in the chekcpoint directory' + \
            f"This is not an existing directory :: {runDir}"
        return 12, cla_err_msg, None
    
    if not os.path.isdir( r'/'.join(opfile.split(r'/')[:-1]) + r'/' ):
        return 14, f'Error in CLA parameter "-op_res_file" : Expected an existing folder in the path specified before the filename', None
    
    ## not required as handled by argparser - will be an integer from 100 to 500
    # if sg_len.isnumeric():
    #     sg_len = int(sg_len)
    # else:
    #     return 16, f'Error in CLA parameter "-len" : Expected an integer value', None
    
    try:
        sg_temp = float(sg_temp)
    except ValueError:
        return 18, f'Error in CLA parameter "-temp" : Expected a float value from 0.0 to 1.0', None
    
    ## not required as handled by argparser - will be an integer from 1 to 5
    # if sg_nsamp.isnumeric():
    #     sg_nsamp = int(sg_nsamp)
    # else:
    #     return 20, f'Error in CLA parameter "-nsamp" : Expected an integer value', None
    
    ## not required as handled by argparser - will be an integer
    # if sg_batsz.isnumeric():
    #     sg_batsz = int(sg_batsz)
    # else:
    #     return 22, f'Error in CLA parameter "-batsz" : Expected an integer value', None

    ## check for sg_topk not required as handled by argparser - default value is 0 and checks for int
    
    return 0, None, [ipf4stygen, ckptdir, runName, opfile, sg_len, sg_temp, sg_nsamp, sg_batsz, sg_topk, logfileloc]

def execute_control_logic(args):
    
    DEBUG_SWITCH = False
    
    check_cla_rc, check_cla_msg, returned_check_cla_returned_list = check_command_line_args(args)
    if check_cla_rc != 0:
        print(f"\n\nFATAL ERROR: Failed processing command line arguments with RC = {check_cla_rc}.\nMessage = {check_cla_msg}\n")
        print(f"\nExiting with Return Code = 10\n")
        exit(10)

    ## show CLA being used
    print(f"\nRunning with command line args as follows:")    
    cla_names = ('-ipf4stygen', '-ckptdir', '-run', '-op_res_file', '-len', '-temp', '-nsamp', '-batsz', '-topk', '-logfileloc')
    for idx, (cla_name, cla_value) in enumerate(zip(cla_names, returned_check_cla_returned_list)):
        print(f"""{idx+1}) {cla_name} = {''.join([ '"', str(cla_value), '"' ])}""")
    
    ipf4stygen, ckptdir, runName, opfile, sg_len, sg_temp, sg_nsamp, sg_batsz, sg_topk, logfileloc = returned_check_cla_returned_list
    del check_cla_rc, check_cla_msg, returned_check_cla_returned_list
    
    ## setup logging file -   levels are DEBUG , INFO , WARNING , ERROR , CRITICAL
    logging.basicConfig(level=logging.INFO, filename=logfileloc,                               \
        filemode='w', format='LOG_LEVEL %(levelname)s : %(asctime)s :: %(message)s')
    
    #######################################################################
    ## STAGE 7 :: STORY GENERATION LOGIC
    #######################################################################
    ## get stories using saved story gen gpt-2 model
    myStr = "\n".join([
        f"\n\n-------------------------------------------------------------------------------",
        f"-------------------------------------------------------------------------------",
        f"    STAGE 7)  STARTING EXECUTION OF STORY GENERATION                           ",
        f"-------------------------------------------------------------------------------",
        f"-------------------------------------------------------------------------------\n\n"
        ])
    print_and_log(myStr, "info")
    myStr = None

    sty_gen_inference_logic_RC, sty_gen_inference_logic_msg, sty_gen_inference_module_results = sty_gen_inference_functionality(ipf4stygen, ckptdir, runName, opfile, sg_len, sg_temp, sg_nsamp, sg_batsz, sg_topk, DEBUG_SWITCH)
    myStr = "\n".join([
        f"\nAfter STORY GENERATION logic execution:",
        f"sty_gen_inference_logic_RC = {sty_gen_inference_logic_RC}",
        f"sty_gen_inference_logic_msg =\n{sty_gen_inference_logic_msg}",
        f"sty_gen_inference_module_results =\n{sty_gen_inference_module_results}\n"
        ])
    print_and_log(myStr, "info")
    myStr = None

    return

if __name__ == '__main__':
    args = argparser.parse_args()
    execute_control_logic(args)

    myStr = f"\n\n\nNormal exit from program.\n"
    print_and_log(myStr, "info")
    myStr = None

    exit(0)
