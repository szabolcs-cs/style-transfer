import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim.python.slim.nets.vgg as vgg
from scipy.optimize import minimize

path_to_vgg19 = "vgg_19.ckpt"  # Download from http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz

STYLE_LAYERS = {'vgg_19/conv1/conv1_2/BiasAdd:0': 5e+2, #Name of layer with associated weight
                'vgg_19/conv2/conv2_2/BiasAdd:0': 5e+2,
                'vgg_19/conv3/conv3_4/BiasAdd:0': 5e+2,
                'vgg_19/conv4/conv4_4/BiasAdd:0': 5e+2,
                'vgg_19/conv5/conv5_4/BiasAdd:0': 5e+2}
CONTENT_LAYER = 'vgg_19/conv4/conv4_2/BiasAdd:0'

minfn_args = {"method": "L-BFGS-B", "jac": True, "bounds": [[0, 255]], #options for the minimizer
              "options": {"factr":0, "ftol": 0,"gtol": 1e-07, "maxcor": 64, "maxiter": 16384, "disp": True}}


def getLayer(L): 
    '''Returns tensor of the specified name from the default graph'''
    return tf.get_default_graph().get_tensor_by_name(L)


def transferStyle(content_image, style_image, tv_loss = 10, content_loss = 0.0002):
    image_ = tf.placeholder(tf.float32, shape=[None, None, 3])  # placeholder for input image
    with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
        vgg.vgg_19([image_])  # construct VGG19 with the image as input
    loss = tv_loss * tf.reduce_mean(tf.image.total_variation(image_)) / tf.reduce_prod(tf.to_float(tf.shape(image_)[:2]))  # total variation loss
    content_in_ = tf.placeholder(tf.float32, getLayer(CONTENT_LAYER).get_shape())  # placeholder for the content features
    loss += content_loss * tf.reduce_mean(tf.square(content_in_ - getLayer(CONTENT_LAYER)))  # content loss
    grams_in_, grams_ = {}, {}  # dictionary for imported and exported style features: layer name -> Tensor
    for layer in STYLE_LAYERS:
        activations = tf.get_default_graph().get_tensor_by_name(layer)#an activation tensor used for tyle loss
        activationsX = activations[:, 1:, 1:, :] - activations[:, :-1, 1:, :]#X gradient of the feature map (convolution by [-1, 1])
        activationsY = activations[:, 1:, 1:, :] - activations[:, 1:, :-1, :]#Y gradient of the feature map (convolution by [-1, 1]^T)
        activations = tf.concat([activations[:, 1:, 1:, :], activationsX, activationsY], 3)#appending gradients to feature map
        grams_[layer] = tf.einsum('ijkl,ijkm->ilm', activations, activations)  # Gram matrix calculation (gram matrices exported here)
        grams_in_[layer] = tf.placeholder(tf.float32, shape=grams_[layer].get_shape())  # gram matrices imported here
        grams_[layer] /= 1e-6 * tf.sqrt(tf.reduce_sum(tf.square(grams_[layer])))  # Normalize gram matrix (multiplication for numerical stability)
        diff = grams_[layer] - grams_in_[layer]  # style difference for this style layer
        loss += STYLE_LAYERS[layer] * tf.log(10000 + tf.reduce_mean(tf.square(diff)))  # accumulate style loss, use log to balance gradients
    loss = tf.to_double(loss) #loss read from here
    image_grad_ = tf.to_double(tf.contrib.layers.flatten(tf.gradients(loss, image_)))  # gradients of the loss read from here
    with tf.Session() as sess:
        tf.train.Saver().restore(sess, path_to_vgg19)  # load VGG parameters from .ckpt file
        content_image = tf.to_float(tf.image.decode_png(tf.read_file(content_image))).eval()  # read content image from disk
        content = sess.run(getLayer(CONTENT_LAYER), {image_: content_image})  # precompute content features
        image_in = tf.image.decode_image(tf.read_file(style_image)).eval()  # read style image from disk
        styles = sess.run({layer: grams_[layer] for layer in STYLE_LAYERS}, {image_: image_in})  # precompute style features
        style_dict = {grams_in_[layer]: styles[layer] for layer in STYLE_LAYERS}  # add precomputed style features to optimization input

        def dfdx(image):  # function passed to L-BFGS that returns the loss and its gradient
            return sess.run([loss, image_grad_], {**style_dict, content_in_: content, image_: np.reshape(image, content_image.shape)})
        minfn_args["bounds"] *= np.array(content_image).size#set the bounds [0, 255] for each pixel for the optimizer
        image = minimize(dfdx, np.float64(np.reshape(content_image, -1)), **minfn_args).x # run the minimizer
        sess.run(tf.write_file("result.png", tf.image.encode_png(tf.reshape(tf.saturate_cast(image, tf.uint8), np.array(content_image).shape))))#write result to disk


if __name__ == "__main__":
    transferStyle(content_image="content_image.png", style_image="style_image.jpg")


