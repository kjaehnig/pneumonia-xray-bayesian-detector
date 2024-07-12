# pneumonia-xray-bayesian-detector

Repository for streamlit hosted web app that will host a bayesian convolutional neural network trained to find pneumonia in chest xrays using Tensorflow-probability. This app hosts all 624 images in the test dataset, as well as the latest trained model parameters.

This app contains a number of options in the sidebar that modify test images: 

* Horizontal / Vertical image flips
* Direct modification of image contrast
* Direct modification of image brightness

There are also sliders and a dropdown list to select different types of image noise to add to the image before performing classification. The five images below illustrate the unmodified and noise-added images.

|:-:|:-:|:-:|:-:|:-:|
|![Unmodified](pneumonia-webapp-unmodified.png)|![Normal Noise](pneumonia-webapp-normal-noise.png)|![Poisson Noise](pneumonia-webapp-poisson-noise.png)|![Uniform Noise](pneumonia-webapp-uniform-noise.png)|![Salt & Pepper Noise](pneumonia-webapp-s&p-noise.png)


<!-- 
| <a style="color:green"><b>True Negative</b></a>  | <a style="color:red"><b>False Positive</b></a> |
|![TN](/images/pneumonia_cxr_nn/cxr_tn.png)|![FP](/images/pneumonia_cxr_nn/cxr_fp.png)|
| <a style="color:red"><b>False Negative</b></a> | <a style="color:green"><b>True Positive</b></a>  |
|![FN](/images/pneumonia_cxr_nn/cxr_fn.png)|![TP](/images/pneumonia_cxr_nn/cxr_tp.png)|

 -->