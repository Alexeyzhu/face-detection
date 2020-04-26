# Face recognizer

Test task for Summer Internship 2020 in VTF Solutions.
 
Face recognition Flask application that performs face detection, 
extraction of face embeddings from each face using deep learning,
training a face recognition model on the embeddings, 
and then finally recognizes faces on video streams from web camera with OpenCV

## About
#### Dataset
The dataset that used to train system contains three people:
* **Arnold Schwarzenegger**
* **George W. Bush**
* **Unknown**, which is used to represent faces of 
people system do not know and wish to label as such

Each class contains a total of 40 images.

#### Project structure
Project has four directories in the root folder:

* *dataset* contains face images organized into subfolders by name
* *facerecognizer* contains files for training, detecting and recognizing faces
* *pretrained_models* contains a pre-trained Caffe deep learning model 
provided by OpenCV to detect faces and a Torch deep learning model which 
produces the 128-D facial embeddings
* *templates* contains html templates for Flask


#### Pipeline

* **OpenCV Caffe-based face detector** to apply face detection, which detects the presence and 
location of a face in an image, but does not identify it. 
It is quite simple to use and efficient

* **Deep learning PyTorch-based model from [OpenFace project](https://cmusatyalab.github.io/openface/)** to extract the 128-d feature vectors (embeddings) 
that quantify each face in an image. 
It is implementation of [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf)
Main   benefit   is   representational   efficiency:   can   achieve state-of-the-art  performance  
(record  99.63%  accuracy  on  LFW, 95.12% on Youtube Faces DB) using only 128-bytes per face.

* **Scikit-learn SVM model** to actually recognize a person trained on embeddings.
It has good performance and simple to train

#### Database structure
In order to store information about persons and embeddings of faces 
a simple and easy **Sqlite** was chosen

Database scheme:

* Face
    * embedding - face embeddings
    * person_id
    
* Person
    * name
 
* Presence - model to track presence of a person in front of a camera
    * person_id
    * timestamp_first 
    * timestamp_last
## How to run
#### Prerequisites
* [Python](https://www.python.org/downloads/)  3.6 and higher
* [Git](https://git-scm.com/downloads)

#### Clone project and setup

* Clone repository

```bash
   git clone https://github.com/Alexeyzhu/.git
```

* Install **requirements.txt**
```bash
   pip install -r requirements.txt
```

You can face problems with installing dlib on Windows. In this case, try to install [Cmake](https://cmake.org/install/) firstly

#### Run program

You can run Flask application with default IP and port
```bash
   python webstreaming.py
```
or specify IP and port that differ from default
```bash
   python webstreaming.py --ip "desired ip" --port port_number
```

