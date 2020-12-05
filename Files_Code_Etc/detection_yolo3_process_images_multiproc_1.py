## -----------------------------------------------------------------------------
## References used in the project:
## Webiste: https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/
## Github: On 05.06.2020, forked from https://github.com/jbrownlee/keras-yolo3 to https://github.com/rbewoor/keras-yolo3
## -----------------------------------------------------------------------------
### Goal: Load the pre-trained Keras model of YOLOv3. Using multiprocessing logic, present new images to identify the objects.
##       Capture the info about which image file has what objects with what score in a multidimensional array.
##       Each job creates its own output file with this information.
## 
##       Information layout example.
##          E.g. suppose img1.jpg has two objects detected: 1st object called label1 with confidence score of label1_score.
##               2nd object called label2 with score of label2_score.
##               Similarly another image img2.jpg has two objects label3 and label4 with respective scores.
##         The data structure will be as below:
##             We want to track these images with a tag as "coco80".
##             [{"img": "img1.jpg", "datasource": "coco80", "det": [["label1", "label1_score"], ["label2", "label2_score"] ] } ,
##              {"img": "img2.jpg", "datasource": "coco80", "det": [["label3", "label3_score"], ["label4", "label4_score"] ] } ]
## 
##         This data array will be used to update a Neo4j database later with the schema of:
##             (:Image{name, dataset}) - HAS{score} -> (:Object{name})
##             dataset property is to track the source of the image:
##             e.g. "coco_val_2017" for images from COCO validation 2017 dataset
##             e.g. "coco_test_2017" for images from COCO test 2017 dataset
##             e.g. "flickr30" for images from Flickr 30,000 images dataset
## -----------------------------------------------------------------------------
## Command line arguments:
## 1) -w            : weights parameter
##                    pre-trained darknet model weights file location
## 2) -if           : images folder parameter
##                    source directory for the images
## 3) -isrc         : image source parameter
##                    tag to keep track of the dataset from where the image was obtained (will be used as a property of the Image node in the Neo4j db)
## 4) -sf           : status frequency paramter
##                    after every how many images processing, should the status be shown
## 5) -nipt         : number images per task paramter
##                    how many images to be processed by each job
## 6) -opfilelocneo : output file location for neo4j
## -----------------------------------------------------------------------------
## Usage example:
##    python3 detection_yolo3_process_images_multiproc_1.py -smp /home/rohit/PyWDUbuntu/thesis/saved_keras_model/yolov3.saved.model   
##       -if /home/rohit/PyWDUbuntu/thesis/Imgs2Detect_more -isrc coco80 -sf 3 -nipt 20 -opfilelocneo /home/rohit/PyWDUbuntu/thesis/Imgs2Detect_more_op4neo
## -----------------------------------------------------------------------------

import argparse
import os
import numpy as np
import struct
import cv2
import json
import sys
import multiprocessing

## changed np.set_printoptions as per https://github.com/numpy/numpy/issues/12987
## was getting error
## ValueError: threshold must be non-NAN, try sys.maxsize for untruncated representation
#np.set_printoptions(threshold=np.nan)
np.set_printoptions(2**31-1)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='test yolov3 network with coco weights')

argparser.add_argument(
    '-smp',
    '--savedmodelpath',
    help='path to the saved kears model of yolov3')

argparser.add_argument(
    '-if',
    '--imagefolder',
    help='path to images folder')

argparser.add_argument(
    '-sf',
    '--statusfrequency',
    type=int,
    help='show status every how many images')

argparser.add_argument(
    '-isrc',
    '--imgsource',
    help='image dataset source')

argparser.add_argument(
    '-opfilelocneo',
    '--ouputfilelocationneo',
    help='location where each worker should dump its neo4j array contents to file')

argparser.add_argument(
    '-nipt',
    '--numberimagespertask',
    type=int,
    help='number of image files per task')

class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print(f"Exiting process {proc_name}")
                self.task_queue.task_done()
                break
            print(f"Process {proc_name} picked job = {next_task.job_num}")
            answer = next_task(proc_name)
            self.task_queue.task_done()
            self.result_queue.put(answer)

class Task(object):
    def __init__(self, _job_num, _data_for_job):
        self.job_num = _job_num
        self.data = _data_for_job
        self.job_proc_name = None  # will be set correctly at run time when consumer picks up task
    def __call__(self, _proc_name):
        self.job_proc_name = _proc_name
        return self.do_everything()
    def __str__(self):
        pass
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
                        
    def draw_boxes_and_make_entry_for_neo(self, image, boxes, labels, obj_thresh, _neo_data_arr_det_info):
        for box in boxes:
            label_str = ''
            label = -1
            
            for i in range(len(labels)):
                if box.classes[i] > obj_thresh:
                    label_str += labels[i]
                    label = i
                    print(labels[i] + ': ' + str(box.classes[i]*100) + '%')
                    # make the entry in data structure for objects and their scores for current image
                    _neo_data_arr_det_info.append([labels[i], round(box.classes[i]*100, 2) ])
                    
            if label >= 0:
                cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), (0,255,0), 3)
                cv2.putText(image, 
                            label_str + ' ' + str(box.get_score()), 
                            (box.xmin, box.ymin - 13), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1e-3 * image.shape[0], 
                            (0,255,0), 2)
        return image
                        
    def make_entry_for_neo(self, boxes, labels, obj_thresh, _neo_data_arr_det_info):
        for box in boxes:
            label_str = ''
            label = -1
            
            for i in range(len(labels)):
                if box.classes[i] > obj_thresh:
                    label_str += labels[i]
                    label = i
                    # make the entry in data structure for objects and their scores for current image
                    _neo_data_arr_det_info.append([labels[i], round(box.classes[i]*100, 2) ])
        return
    
    def do_everything(self):
        from keras.models import Model, load_model

        #saved_model_location = self.data[0]
        #image_files_arr_job  = self.data[1]
        #img_dataset          = self.data[2]
        #opfilelocneo         = self.data[3]
        #print_status_freq    = self.data[4]
        saved_model_location, image_files_arr_job, img_dataset , opfilelocneo, print_status_freq = self.data
        
        # set some parameters for network
        net_h, net_w = 416, 416
        obj_thresh, nms_thresh = 0.5, 0.45
        anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
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

        # DEBUGGING ONLY
        #print(f"\nProcess = {self.job_proc_name}, job = {self.job_num}\nData for the job:\nimage_files_arr_job:\n{image_files_arr_job}\nimg_dataset = {img_dataset} , opfilelocneo = {opfilelocneo}\n\n")
        # DEBUGGING ONLY

        # Info about objects detected to be stored in a multi-dimensional array.
        # E.g. suppose img1.jpg has two objects detected: 1st object called label1 with confidence score of label1_score.
        #      2nd object called label2 with score of label2_score.
        #      Similarly another image img2.jpg has two objects label3 and label4 with respective scores.
        #      The data structure will be as below.
        #      We want to track these images with a tag as "coco80".
        # [{"img": "img1.jpg", "datasource": "coco80", "det": [["label1", "label1_score"], ["label2", "label2_score"] ] } ,
        #  {"img": "img2.jpg", "datasource": "coco80", "det": [["label3", "label3_score"], ["label4", "label4_score"] ] } ]
        # Blank array setup once
        job_neo_data = []
        
        file_skipped_count = 0
        # process each input image file from array
        job_img_files_count = len(image_files_arr_job)

        # reload the saved model
        reloaded_yolo_model = load_model(saved_model_location)

        for file_no, each_image in enumerate(image_files_arr_job):
            if file_no % print_status_freq == 0:
                print(f"\n\n{self.job_proc_name}-Job-{self.job_num}:: Processing image {file_no +1} of {job_img_files_count} : {os.path.basename(each_image)}")
            
            # preprocess the image
            image = cv2.imread(each_image)
            image_h, image_w, _ = image.shape
            try:
                new_image = self.preprocess_input(image, net_h, net_w)
            except Exception as error_image_resize:
                #print(f"\n\nFile skipped file - ERROR:\n{error_image_resize}\n")
                file_skipped_count += 1
                continue   # proceed to next image
            
            # run the prediction
            yolos = reloaded_yolo_model.predict(new_image)
            boxes = []

            for i in range(len(yolos)):
                # decode the output of the network
                boxes += self.decode_netout(yolos[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)
            
            # correct the sizes of the bounding boxes
            self.correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

            # suppress non-maximal boxes
            self.do_nms(boxes, nms_thresh)

            # create entry for this image in neo4j data array
            job_neo_data.append({"img": os.path.basename(each_image) , "datasource": img_dataset , "det": [] } )

            # IMPORTANT: uncomment to draw bouding box on image and save it to disk
            # draw bounding boxes on the image using labels. Also add the detection data for this image to the neo4j array
            #self.draw_boxes_and_make_entry_for_neo(image, boxes, labels, obj_thresh, job_neo_data[-1]["det"])
            # write the image with bounding boxes to file
            #cv2.imwrite(each_image[:-4] + '_det' + each_image[-4:], (image).astype('uint8'))
            
            # add the detection data for this image to the neo4j array. No drawing of bounding boxes
            self.make_entry_for_neo(boxes, labels, obj_thresh, job_neo_data[-1]["det"])
        
        # write the neo4j array data to a file. But if there is no data then set the file as None
        try:
            if len(job_neo_data) == 0:
                job_neo_arr_filename = None
            else:
                job_neo_arr_filename = opfilelocneo + r"job_" + str(self.job_num) + r".txt"
                with open(job_neo_arr_filename, "w") as opfile:
                    json.dump(job_neo_data, opfile)
            print(f"\n\n{self.job_proc_name}-Job-{self.job_num}:: Completed. Output file: {job_neo_arr_filename}")
            job_rc = 0
        except Exception as error_job_neo_data_write:
            print(f"\n\nJob {self.job_num} had problem writing neo data to file\nERROR:\n{error_job_neo_data_write}")
            print(f"Set Job RC = 500 and the output file as None")
            job_rc = 500
            job_neo_arr_filename = None
        return [self.job_num, job_rc, file_skipped_count, job_neo_arr_filename]

def _main_(args):
    # process command line arguments
    savedmodelpath = args.savedmodelpath       # -smp parameter, location of the saved kears model for pre-trained yolov3 model
    image_path     = args.imagefolder          # -if parameter, where to pick the images from to process
    img_dataset    = args.imgsource            # -isrc parameter, image node property value for dataset
    status_freq    = args.statusfrequency      # -sf parameter, after how many images info is processed for neo4j inserts should a status message be shown
    nipt           = args.numberimagespertask  # -nipt parameter, how many images files to be processed in each task
    opfilelocneo   = args.ouputfilelocationneo # -opfileneo parameter, location where dump of neo4j array of each task should be written to file

    # check the status_frequency is valid
    if status_freq < 1:
        print(f"FATAL ERROR: status_frequency argument must be a a non-zero whole number.\nExiting with RC=100")
        exit(100)
    
    # check valid input for the -nipt parameter
    if nipt < 1:
        print(f"FATAL ERROR: nipt argument must be a a non-zero whole number.\nExiting with RC=105")
        exit(105)

    # setup output directory if not existing
    try:
        if opfilelocneo[-1] != r"/":
            opfilelocneo += r"/"
        if not os.path.isdir(opfilelocneo):
            os.makedirs(opfilelocneo) # does not exist, make it
    except:
        print(f"FATAL ERROR: Problem creating output directory.\nExiting with RC=110")
        exit(110)
    
    # find all the files in the image path folder and create a list of absolute path and filename.
    # -- assuming ALL files in the specified folder are jpg's to be processed
    image_files_arr = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
    img_files_count = len(image_files_arr)
    
    print(f"\n\nImages folder: {image_path}\nImages found count = {img_files_count}\nImage source = {img_dataset}")
    # DEBUGGING ONLY
    #exit(0)
    # DEBUGGING ONLY

    # calculate number of jobs required
    num_jobs = int(len(image_files_arr) / nipt + ( (len(image_files_arr) % nipt)!=0 ) * 1)
    print(f"With nipt = {nipt}, number of jobs required = {num_jobs}\n")

    # DEBUGGING ONLY
    #print(f"Image array FULL before division:\n{image_files_arr}\n")
    # DEBUGGING ONLY

    # Establish communication queues
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()

    # start the consumers
    #num_consumers = multiprocessing.cpu_count() - 1
    num_consumers = 8
    consumers = [ Consumer(tasks, results) for i in range(num_consumers) ]
    for w in consumers:
        w.start()
    print(f"Created {num_consumers} consumers")

    # enqueue the jobs
    #     data for each job is an array of three things:
    #        location of saved model
    #        image slice array
    #        dataset name e.g. coco80
    #        output directory
    #        print to console status frequency
    for i in range(num_jobs):
        data_for_job = []
        data_for_job.append(savedmodelpath)
        if i == (num_jobs - 1):
            data_for_job.append(image_files_arr[(num_jobs-1)*nipt : ])
        else:
            data_for_job.append(image_files_arr[i*nipt : (i+1)*nipt])
        data_for_job.append(img_dataset)
        data_for_job.append(opfilelocneo)
        data_for_job.append(status_freq)
        tasks.put(Task(i+1, data_for_job))
    
    # Add a poison pill for each consumer
    for i in range(num_consumers):
        tasks.put(None)

    # Wait for all of the tasks to finish
    tasks.join()
    print(f"\n\n------------  All consumers rejoined  ------------\n\n")
    
    # Process the results
    #    track the skipped files counts
    #    track the return codes of all the jobs
    #    extract the output file created (if any)
    total_files_skipped = 0
    all_jobs_rc = 0
    op_files_for_neo = []
    for i in range(num_jobs):
        job_number, job_rc, job_file_skipped_count, job_op_file = results.get()
        if job_rc != 0:
            all_jobs_rc = 1
        total_files_skipped += job_file_skipped_count
        print(f"Job {job_number}, return code = {job_rc}, skipped files = {job_file_skipped_count}, output_file = {job_op_file}")
        if job_op_file is not None:
            op_files_for_neo.append(job_op_file)
    
    # show summary data
    print(f"\n\n-------- SUMMARY ------\nImages folder: {image_path}\nImage source = {img_dataset}\nTotal images found count = {img_files_count}")
    print(f"Total files skipped across all jobs = {total_files_skipped}\n\nTotal files processed = {img_files_count - total_files_skipped}")
    print(f"{'ALL ' if all_jobs_rc == 0 else 'ONLY SOME '} jobs completed with RC=0")
    print(f"Total output files created = {len(op_files_for_neo)}")
    print(f"The output files:")
    for each_op_file in op_files_for_neo:
        print(f"\t{each_op_file}")

    print(f"\n\n\nNormal exit from program.\n")

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)