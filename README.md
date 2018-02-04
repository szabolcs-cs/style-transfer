# style-transfer
This is an efficient, high quality implementation of [neural style transfer](https://arxiv.org/abs/1508.06576) in Tensorflow, adding some tweaks of my own.
## Prerequisites
* Tensorflow (tested on 1.4)
* The VGG-19 network from [this](https://github.com/tensorflow/models/tree/master/research/slim) page.
## Getting started
* Download files to a local folder
* Install Python 3.6 from [here](https://www.python.org/downloads/)
  * Windows users use [64-bit version](https://www.python.org/downloads/windows/)
* [Install Tensorflow](https://www.tensorflow.org/install/) (GPU version recommended if you have an nvidia GPU)
* Extract the [file](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz) containing VGG-19 
* Edit style-transfer.py to point the ```path_to_vgg19``` variable to ```vgg_19.ckpt``` from the above file
* You can change the style and content image used by changing the ```content_image``` and ```style_image``` parameters of the ```transferStyle``` function
* You can run the script by running ```python3 style-transfer.py``` from the command line
  * This will produce ```result.png``` with the same resolution as the content image

## Parameters
* The ```tv_loss``` parameter controls the amount of *smoothing* applied during style transfer. Setting this to 0 can produce more ringing artefacts
* The ```content_loss``` parameter controls how strongly the contents of the *content image* should be enforced.
* The ```maxiter``` parameter of the optimizer controls the maximum number of iterations to run. More iterations generally mean higher quality. For best results set this to a high number (e.g. 50000) and let the optimizer run until convergence (which should happen before 50000 iterations are reached)
* The ```maxcor``` parameter controls the amount of information tracked by the L-BFGS optimizer about the curvature of the function being minimized. A higher number should mean a higher quality output, but also results in slower iteration time (The optimizer runs on the CPU)

## Examples
Using the following image as the style image:

![Style image](style_image.jpg)

And this one as the content image:

![Content image](content_image.png)

Produces:

![Result](result.png)

## Additional style transfer tweaks used
* Added total variation loss used in some other implementations
* Added *x* and *y* gradients of feature maps to Gram matrix generation for richer style representation
* More equal balancing of style loss gradients
Details below:

### Effects of feature gradients on style representations
It's possible to make style representations obtaned from the style image richer by also including the horizontal and vertical gradients of the activation tensors in the Gram matrix. The following line reads in the activation tensor:
```
activations = getLayer(layer)
```
The following 2 lines calculate the *x* and *y* gradients of the feature maps:
```
activationsX = activations[:, 1:, 1:, :] - activations[:, :-1, 1:, :]
activationsY = activations[:, 1:, 1:, :] - activations[:, 1:, :-1, :]
```
Which is equivalent to convolving the feature maps by the [-1, 1] convolution kernel horizontally and vertically. Then these features are concatenated with the original activations to effectively triple the number of features available to construct the Gram matrix:
```
activations = tf.concat([activations[:, 1:, 1:, :], activationsX, activationsY], 3)
```
This results in a more consistent representation of style
#### Without feature gradients
![Without feature gradients](result_nograd.png)
#### With feature gradients
![With feature gradients](result_withgrad.png)

## Balancing style gradients during minimization
The original style loss as defined in ![Gatys et. al.](https://arxiv.org/abs/1508.06576) is defined as follows:

![\mathcal{L}_{style}(\vec{a},\vec{x})=\sum_{l=0}^{L}w_lE_l](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7Bstyle%7D%28%5Cvec%7Ba%7D%2C%5Cvec%7Bx%7D%29%3D%5Csum_%7Bl%3D0%7D%5E%7BL%7Dw_lE_l)

One problem with this function is that the relative magnitude of gradients of the style loss from each activation layer can change signicifantly during optimization. This is combated by instead minimizing the sum of the logarithms of the style loss for each layer:

![\mathcal{L}_{style}(\vec{a},\vec{x})=\sum_{l=0}^{L}w_l\log(E_l)](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7Bstyle%7D%28%5Cvec%7Ba%7D%2C%5Cvec%7Bx%7D%29%3D%5Csum_%7Bl%3D0%7D%5E%7BL%7Dw_l%5Clog%28E_l%29)


## Troubleshooting, tips and tricks
* If you get an "out of memory" error message try to reduce the resolution of the content image
* It's possible that some of the ringing artefacts in the result are produced by JPEG artefacts in the style image. This can be mitigated by using an uncompressed style image, or one that was resized from a jpeg image to a smaller resolution and then saved as an uncompressed image.
* The best way to capture a rich style representation from an image is to find a high resultion version of it, scale it down with a good quality filter to a lower resolution (something around 512 to 1024 pixels on the long side) and save with lossless compression.

## Future work
* Apply loss at multiple levels of an image pyramid
* Arbitrary resolution of the output
