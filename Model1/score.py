#INPUT:image_b64:string
#OUTPUT:y_json:string
#FUNCTION:score
from tensorflow.python.saved_model import tag_constants
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

MODEL_DIR = '/tsmc_model/model_files'
version = 1
import_path = os.path.join(MODEL_DIR, str(version))

graph = tf.Graph()
with graph.as_default():
    with tf.Session(graph=graph) as sess:
        meta_graph_def = tf.saved_model.loader.load(
           sess,
           [tag_constants.SERVING],
           import_path
        )
        for op in sess.graph.get_operations():
            print(op.name)
        x = sess.graph.get_tensor_by_name('Conv1_input_5:0')
        y = sess.graph.get_tensor_by_name('Softmax_5/Softmax:0')
        y_out = sess.run(y, feed_dict = {x: [test_images[0]]})
        y_json = np.argmax(y_out[0])
        print(y_json)
