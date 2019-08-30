#INPUT:image_b64:string
#OUTPUT:y_json:string
#FUNCTION:score
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
test_images = test_images / 255.0
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
new_model = tf.keras.models.load_model('/tsmc_model/model_files/my_model.h5')
predictions = new_model.predict(test_images)
y_json = str(np.argmax(predictions[0]))
