# TensorFlow - Help Protect the Great Barrier Reef

### Goal of the Competition
The goal of this competition is to accurately identify starfish in real-time by building an object detection model trained on underwater videos of coral reefs. [[Competition Homepage](https://www.kaggle.com/c/tensorflow-great-barrier-reef/overview)]

### Model Submission & Scores

|        Models       | Train Image Size | Inference Image Size| Post-processing | Public LB Score | Private LB Score |
|        :----:       |   :--------:     |     :--------:      |  :--------:     |  :-------:      |     :-------:    |
|       Yolov5x       |    1600 x 1600   |       1920 x 1920   |        -        |      0.615      |        0.644     |
|       Yolov5x       |    1600 x 1600   |       1920 x 1920   |      Tracking   |      0.633      |        0.618     |
|       Yolov5s6      |    1920 x 1920   |       1920 x 1920   |        -        |      0.564      |        0.581     |
|     Yolov5s6-p7     |    1536 x 1536   |       1536 x 1536   |        -        |      0.474      |        0.527     |

### Summary

#### Cross-Validation
The cross-validation is done by splitting the data into 5 folds based on the subsequences (gap-free subset of given videos), and the model is trained to optimize F2 score and evaluated on F2 score, together with visualizing the performance of the models on unseen video subsequences.

#### Models
The final solution with the highest score is a single Yolov5x model, without any post-processing and the bounding boxes were filtered by using confidence threshold of 0.32 and IoU threshold of 0.40. Other trials are made by using EfficientDet, YoloV5s, Yolov5m, Yolov5l, and YoloX, but the performance didn't outperform Yolov5x with various amount of parameter tuning.

#### Data Augmentation
* Mosaic (prob@1.0 to prob@0.5 boost the performance around 0.13)
* Mixup
* Random Scaling
* Color Jitter
* Translate
* Flip (Up down & Left right)
* Median Blur
* CLAHE
* Random Brightness & Contrast
* Gaussian Noise

#### Optimizer

* SGD, with initial learning rate of 0.1, final OneCycleLR learning rate of 0.1  
* Adam, with initial learning rate of 0.001, final OneCycleLR learning rate of 0.1 <strong>(used in the final model)</strong>  
* AdamW, with initial learning rate of 0.001, final OneCycleLR learning rate of 0.1  

#### Training
After numerous of trials, the process can be summarized as following:
* Some data augmentation can impact the Public LB score significantly (scaling, mosaic, rotation, HSV)
* Default model hyperparameter parameters tend to work better
* Larger model tend to work better
* Background images doesn't help (train with images that contain objects only produce higher OOF and LB scores)
* Train on higher image resolution (around 1.25x to 1.75x of the original size) yields better Public LB score
* Inference on higher image resolution sometimes yield better Public LB score

#### Inference
In the early stage of the competition, I found that for some models, altering the confidence and IoU threshold for inference can improve the Public LB score around 0.01 to 0.03. Also, by applying tracking, the LB score can be improved around 0.008. After [the post](https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/300638) indicated that increasing the image size (range from 3,600 to 10,000) can bring a boost in Public LB score. I ran the model on images with size increased from 1.25x to 2.0x, and obtained the best performance for increasing image reference size around 1.25x. Over 2.0x, from my perspective, is unpractical in terms of speed and memory needs.

#### Things tried but didn't work well
* Weighted Box Fusion (WBF)
* Tracking
* Background Images

#### Things in mind but haven't taken into practice
* Increase dataset size by generating "fake COTS" using GANs
* Train a classifier as the second stage in the inference pipeline to perform binary classification (COTs or Not COTs) on image cropped by bounding boxes
* Fix some wired bounding box in the training images
* In unlabelled images, there are actually some objects exist, should label those and increase image size
* Using cut and paste, randomly paste objects to background images and use that for training
* Try more types of model and do better ensemble
* Put images that have high FPs as background images

#### Performance Analysis
After calculating the OOF score and visualizing the detection results on unseen subsequences, I noticed that my model was suffering from False Positives (FPs). I tried to cope with this issue by adding background images and playing around with confidence threshold, but none of these improve the scores. Hence, I think it is probably because the background image that I added didn't really help the model to learn. I also altered the hyperparameter to put more weight on objectness loss, but without getting any improvement.

#### Lesson Learned
After reading some of the winning solutions, I noticed that I did catch the tail of things that can help me improve the model performance but I lost it because I didn't take those into practice. Since I completely relied on Kaggle notebook for 40 hours GPU usage a week, I can only train limit amount of models to verify the idea. If there is any other object detection competition in the future, I would do the following things:
* Put at least 80% of time on data, create a systematic pipeline for data exploration, analysis, visualization, preprocessing, feature engineering, etc.. Also, try the best to find a cross-validation set that can explain the generalization of the model 
* Setup a systematic validation and evaluation pipeline, and keep track any trial and its influence on the model's performance
* Even the idea has a chance of 0.99% to work, try it

### Links & Reference
* Training Notebook: https://www.kaggle.com/sjyangkevin/tf-cots-yolov5-training-pipeline
* Inference Notebook: https://www.kaggle.com/sjyangkevin/great-barrier-reef-yolov5-infer
* YoloV5 (GitHub Repository): https://github.com/ultralytics/yolov5
* YoloX (GitHub Repository): https://github.com/Megvii-BaseDetection/YOLOX
* Norfair - 2D Object Tracking (GitHub Repository): https://github.com/tryolabs/norfair
