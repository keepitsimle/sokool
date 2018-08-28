import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 初始化的效果好,
# 小了逐步传递缩小难以起作用
# 大了发散失效
def xavier_init(fan_in,fan_out,constant=1):

    low = -constant*(np.sqrt(6.0/(fan_in+fan_out)))
    high = -low

    return tf.random_uniform((fan_in,fan_out),
                             minval=fan_in,
                             maxval=fan_out,
                             dtype=tf.float32)

class AdditiveGaussianNoiseAutoEncoder(object):
    def __init__(self,n_input,n_hidden,):
