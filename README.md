# image-blur-detection

### To get all the Dependencies (run the command) after forking / cloning [this](https://github.com/sauravpanchal/image-blur-detection.git) repository :
`pip install requirements.txt`

### Approach - 1 : Convolutional Neural Network (CNN / ConvNets)

(Step-1) Load & Pickle Train dataset (run the command) :

`python train.py`

(Step-2) Load & Pickle Test dataset (run the command) :

`python testdata.py`

A Convolutional Neural Network is trained over [CERTH_ImageBlurDataset](http://mklab.iti.gr/files/imageblur/CERTH_ImageBlurDataset.zip) (~3.7 GB) yielding accuracy of **58.18%** on evaluation dataset.
Accuracy can further be improved by increase input dimensions (of first layer) / model's complexity or tweaking number of epochs.

(Step-3) To train the CNN model (run the command) :

`python CNN.py`

<hr>

### Approach - 2 : Variance of Laplacian

Here we calculate variance of Laplcian; giving value which defines blurry metric. If it's below certain threshold (here it's 435) image can be classified as burry else it is going 
to be non-blurry. This model gave accuracy of around **87.23%**. It's also, performed over the above given dataset only.

(Step-1) To run the script for Laplacian approach (run the command) :

`python Laplacian.py`

<hr>

##### NOTE : While running scripts you can ignore / supress warnings / informations which arises due to difference in API's and other dependencies