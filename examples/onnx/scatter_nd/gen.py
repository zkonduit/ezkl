from torch import nn
import json
import numpy as np
import tf2onnx


import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


# gather_nd in tf then export to onnx
x = in1 = Input((4, 1), dtype=tf.int32)
w = in2 = Input((4, ), dtype=tf.int32)

class MyLayer(Layer):
    def call(self, x, w):
        shape = tf.constant([8])
        return tf.scatter_nd(x, w, shape)

x = MyLayer()(x, w)



tm = Model((in1, in2), x)
tm.summary()
tm.compile(optimizer='adam', loss='mse')

shape = [1, 4, 1]
index_shape = [1, 4]
# After training, export to onnx (network.onnx) and create a data file (input.json)
x = np.random.randint(0, 4, shape)
# w = random int tensor
w = np.random.randint(0, 4, index_shape)

spec = tf.TensorSpec(shape, tf.int32, name='input_0')
index_spec = tf.TensorSpec(index_shape, tf.int32, name='input_1')

model_path = "network.onnx"

tf2onnx.convert.from_keras(tm, input_signature=[spec, index_spec], inputs_as_nchw=['input_0', 'input_1'], opset=12, output_path=model_path)


d = x.reshape([-1]).tolist()
d1 = w.reshape([-1]).tolist()


data = dict(
    input_data=[d, d1],
)

# Serialize data into file:
json.dump(data, open("input.json", 'w'))
