import sys, os
import requests, urllib.request
from io import BytesIO
import time, datetime
from django.core.files.base import ContentFile
from .models import Imageupload
from django.shortcuts import get_object_or_404

import tensorflow as tf
print(tf.__version__)
import keras
import numpy as np
import albumentations as A
import segmentation_models as sm
from PIL import Image
from skimage import io

from tensorflow.python.keras.backend import set_session

# Load the model
BACKBONE = 'efficientnetb4'
CLASSES = ['bg', 'artery', 'ureter', 'e_artery', 'f_tube', 'ovary', 'rl', 'ipl', 'peri', 'scalpel', 'wound']
preprocess_input = sm.get_preprocessing(BACKBONE)
n_classes = len(CLASSES) + 1 # add unlabelled
activation = 'softmax'
### use session and graph for django!!!
gSess = tf.Session()
gGraph = tf.get_default_graph()
set_session(gSess)
gModel = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
#gModel.load_weights(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../model/UNET_8809_84_model.h5'))
gModel.load_weights('model/UNET_8809_84_model.h5')
print("segmentation model loaded!")

# helper function
def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

# TMUH segmentation
def o_segmentation(pk_o):
    photo_o = get_object_or_404(Imageupload, pk=pk_o)
    img_name = photo_o.image_file.url
    # get file name and extension
    f_n = img_name.split("/")[-1].split(".")[0]
    f_e = img_name.split(".")[-1]
    # remove special charactor
    tbd = ['!','@','#','$','%','^','&','*','(',')','-','+','=']
    for i in tbd:
        f_n = f_n.replace(i,'')
    # if the extension is too long make it .png
    if len(f_e) > 7:
        f_e = "png"
    out_f_name = f_n + "_out." + f_e
    # Load the input image
    if "http" in img_name: # for GCS
        response = requests.get(photo_o.image_file.url)
        img = Image.open(BytesIO(response.content))
        img.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../media/images/temp.png'), format='png')
        image = io.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../media/images/temp.png'))
    else:
        image = io.imread(img_name[1:])
    # some settings
    target_width = 640
    target_height = 480
    need_resize = False
    # check if the size is 640*480
    if image.shape[0] != 480 or image.shape[1] != 640:
        # save the original size for later use
        target_width = image.shape[1]
        target_height = image.shape[0]
        need_resize = True
        # Load the input image
        if "http" in img_name: # for GCS
            im = Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../media/images/temp.png'))
            # resize to 640*480
            im = im.resize((640, 480))
            im.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../media/images/temp.png'), format='png')
            image = io.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../media/images/temp.png'))
        else:
            im = Image.open(img_name[1:])
            # resize to 640*480
            im = im.resize((640, 480))
            im.save("media/images/temp.png", format='png')
            image = io.imread("media/images/temp.png")

    image = preprocess_input(image) # use the preprocessing input method based on the backbone you used
    image = np.expand_dims(image, axis = 0)
    # in django predict the model with graph and session
    with gGraph.as_default():
        set_session(gSess)
        pr_mask = gModel.predict(image) # change the shape to (1 H, W)
    pr_mask = pr_mask.squeeze()
    pr_mask = np.argmax(pr_mask, axis = 2)
    # save_color(pr_mask, savedata_path, need_resize, width, height)
    colors = [(0,0,0),      #bg
              (250,0,0),    #artery
              (0,250,0),    #ureter
              (0,0,250),    #e_artery
              (0,250,250),  #f_tube
              (250,0,250),  #ovary
              (100,100,100), #rl
              (150,150,150), #ipl
              (15,205,130),  #peri
              (250,250,0),   #scalpel
              (205,50,130),  #wound
              (0,0,0)]       #unlabel

    height, width = pr_mask.shape[0], pr_mask.shape[1]
    img_mask = Image.new(mode = "RGB", size = (width, height))
    px = img_mask.load()

    for x in range(0,width):
        for y in range(0,height):
             px[x,y] = colors[pr_mask[y][x]]
    # resize if the original size is not 640*480
    if need_resize is True:
        img_mask = img_mask.resize((target_width, target_height), Image.NEAREST)

    # save the result image
    img_io = BytesIO()
    img_mask.save(img_io, format='png')
    img_content = ContentFile(img_io.getvalue(), out_f_name)

    photo_o.result_file = img_content
    photo_o.readiness = "2"
    photo_o.save()
    print("complete ", img_name)
    return photo_o.result_file.url
