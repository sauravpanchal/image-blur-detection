# image-blur-detection

### Approach : Convolutional Neural Network

Load & Pickle Train dataset :

`train.py`

Load & Pickle Test dataset :

`testdata.py`

A Convolutional Neural Network is trained over [CERTH_ImageBlurDataset](http://mklab.iti.gr/files/imageblur/CERTH_ImageBlurDataset.zip) yielding accuracy of **58.18%** on evaluation dataset.
Accuracy can further be improved by increase input dimensions (of first layer) / model's complexity or tweaking number of epochs.

To train the CNN model :

`CNN.py`

