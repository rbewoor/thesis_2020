# thesis_2020
Master thesis work as part of M.Sc. Big Data and Business Analytics coursework over the period June 2020 to November 2020.

This document explains the various files used in the implementation of the master thesis work over the period June-Nov 2020.
Please refer to the coloquium presentation for some high level details.

Goal was to design and implement "Story generation for young children by accepting a voice input description from user".

Implementation done by weaving together multiple neural networks to process the data in stages. One time setup of a Neo4j database containing information about which objects are in which images was required. The pipeline consisted of: a) Perform Speech-to-Text (STT) b) Id key elements c) Retrieve images user selection of images d) Perform Image Captioning e) Selection of images and optional correction of captions f) Generate story.

GUI implemented with Tkinter for a) STT stage: Wav file recording, playback and confirmation  b) Id Key elements stage: display and selection c) Images from database retrieval stage: selection with optional object detection d) Image caption generation stage: view images and captions with optional correction of captions.
No GUI for the final story generation stage.
Implementation in Python3 in Anaconda environments.

The models were: 1) Pre-trained STT DeepSpeech version 0.7.3 (via pip instal of package and downloading the model) 2) Object detection model defined by loading the pre-trained model weights and then saved as a keras model: file "saved_keras_model.tar.gz" 3) Image caption model WITHOUT-attention trained and saved the weights for reloading the model later: file "Decoder_Run_3_Wt_ep_18.h5" 4) Image caption model WITH-attenion trained and saved the checkpoint files for reloading model later: file "Thesis_ImgCapATTENTION_ChkPts_In_Run4_Ep19.tar.gz" 5) GPT-2 model fine-tuned on 11 files of CBT dataset and then saved the checkpoints for reloading later: file "Run2_File11_2_checkpoint_run2.tar.gz"

For the image captioning model, the training was done with MS-COCO 2017 version. Ground truth captions were available in the train and val json annotations files.

Environment setup related.
1) Different Anaconda environments were setup for different parts of the code execution. See file for environment setup: conda_environment_setup_20201111.txt

Scripts and files:

1) Build YOLOv3 object detector model by loading the pre-trained weights and save as a keras model:
	detection_yolov3_build_save_model_1.py
2) Process images and create data structure:
	detection_yolo3_process_images_multiproc_1.py
3) Process data structure and load Neo4j db:
	detection_update_neo_2.py
4) Perform STT to GUI selection of image captioning (except story generation):
	comb_functional_mic_stt_idKey_queryNeo_imgSelect_imgCap_10A.py
5) Perform Story Generation and save data structures with output to file:
	StyGen_Inference_4A.py
6) See results of story generation stage by loading saved data structures from file:
	StyGen_Show_Results_2.ipynb
7) Story generation using only single sentence:
	StyGen_Inf_Trained_Model_3_single_sentences.ipynb
8) Preprocess files to train GPT-2 model:
	StyGen_processFiles_4_training_CBT_1.ipynb
9) Training script for GPT-2 model:
	StyGen_GPT2_train_Run2_1.ipynb
10) Sample of training script for WITHOUT-attention image captioning model:
	ImgCapTraining_Kagg_CocoTrain2017_run3_10_WIP.ipynb
11) Sample of training script for WITH-attention image captioning model:
	ImgCap_ShowAndTell_Kag_Coco2017_run4_1_Ep19to21.ipynb
12) Pickle file, tokenizer used to train WITH-attention model:
	ATTEND_tokenizer_from_training_100k.pkl
13) Pickle file, 5k images set aside for BLEU scoring of WITH-attention model:
	ATTEND_images_5k_for_bleu_scoring.pkl
14) Pickle file, dictionary of WITH-attention model inference outputs on the 5k images and the ground truth captions:
	CompDict_ImgCapATTEND_Run4_Ep21_BleuScore_With_GT.pkl
15) Weights file, WITHOUT-attention model:
	Decoder_Run_3_Wt_ep_18.h5
16) Weights file, object detector downloaded from https://pjreddie.com/media/files/yolov3.weights (not uploaded to Github due to size):
	yoloWeights.tar.gz
17) Saved keras model for object detector reloading (not uploaded to Github due to size):
	saved_keras_model.tar.gz
18) Checkpoint files to reload fine-tuned GPT-2 model (not uploaded to Github due to size):
	Run2_File11_2_checkpoint_run2.tar.gz
19) The 11 files from CBT dataset used to fine-tune medium GPT-2 model:
	all files in folder "CBT_fineTune_GPT2"
20) Checkpoint files to reload with-attention model (not uploaded to Github due to size):
	Thesis_ImgCapATTENTION_ChkPts_In_Run4_Ep19.tar.gz
21) Pickle file, for WITHOUT-attention model, the word-to-index dictionary data structure:
	wordtoix_train_97000.pkl
22) Pickle file, for WITHOUT-attention model, the index-to-word dictionary data structure:
	ixtoword_train_97000.pkl
23) Script to calculate BLEU scores for WITH-attention model using greedy search:
	ImgCapATTEND_Bleu_Greedy_Jpgs_LAPPY_1_Run4_Ep21.ipynb
24) Script to calculate BLEU scores for WITHOUT-attention model using greedy search:
	ImgCap_Bleu_GreedySearch_Encodings_LAPPY_1_Run3_Ep18.ipynb
25) Script to calculate BLEU scores for WITHOUT-attention model using beam search with width=3:
	ImgCap_Bleu_BeamWidth3_Encodings_LAPPY_1_Run3_Ep18.ipynb
26) Script to calculate BLEU scores for WITHOUT-attention model using beam search with width=5:
	ImgCap_Bleu_BeamWidth5_Encodings_LAPPY_1_Run3_Ep18.ipynb
27) Script to view the results of WITH-attention model caption results against some images and their ground truth captions:
	demo_ImgCap_Attend_Reload_Lappy_Infer_Greedy_1.ipynb
28) Script to check the validation losses for WITHOUT-attention model, reloads specified epochs weights file and processes the 3000 images kept aside as validation dataset:
	ImgCap_Check_Val_Loss_Lappy_1.py 
29) Plot of model for decoder of WITHOUT-attention model:
	ImgCap_RNN_Decoder_model3_plot_1.png
30) Save the encodings of the images to avoid bottleneck layer speed issues (for WITHOUT-attention model):
	ImgCapTraining_Train2017_onlyEncoding_1.ipynb
31) Annotations files for the Train and Validation splits of the original COCO2017 dataset (approx. 118k images in Train split and 5k images in Validation split) (not uploaded to Github due to size):
	captions_train2017.json    ,    captions_val2017.json
