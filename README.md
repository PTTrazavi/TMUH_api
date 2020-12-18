# Install h5py to solve .h5 file reading issue.  
h5py==2.10.0  

# Use session and graph to load model once only to make application faster.  
https://github.com/tensorflow/tensorflow/issues/28287  
```python
# outside of the function
from tensorflow.python.keras.backend import set_session
### use session and graph for django!!!
gSess = tf.Session()
gGraph = tf.get_default_graph()
set_session(gSess)
gModel = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
gModel.load_weights('model/UNET_8809_53_model.h5')
print("segmentation model loaded!")

# inside of the function
    with gGraph.as_default():
          set_session(gSess)
          pr_mask = gModel.predict(image)
```
