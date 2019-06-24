# Image Style Transfer using PyTorch
> Combine two images and produce new image with style of first image and content of second image

## Table of contents
* [About Project](#about-project)
* [Languages or Frameworks Used](#languages-or-frameworks-used)
* [Setup](#setup)
* [More Learning](#more-learning)

## About Project:

  ![Output of Model](https://github.com/SurajChinna/Image-Style-Transfer-Pytorch/blob/master/assets/download.png "Output of Model")
  
  The main goal of project is to combine two images and produce new image. The combination works in slightly different way i.e., we       combine the style of one image with the content of other image. First we take the image from which we want to extract content usually     called <b>content image</b> and take another image from which the style is to be extracted usually called **style image**. This is     the implementation of [this](https://arxiv.org/pdf/1508.06576.pdf) research paper.  
  
  Convolutional Neural Networks are a type of neural networks which are used widely in Image classification and recongnition. A CNN architecture called VGG19 has been used in this project. The starting layers in this architecture extract the basic features and shapes and later layers will extract more complex image patterns. So for the output image we will take the **content** from later layers of CNN. For extracting the style of image, we take the correlations between different layers using [Gram Matrix](https://en.wikipedia.org/wiki/Gramian_matrix)
  
 Initially, we take any random image as target(or taking the content image would be useful) and compute the **Content loss** and **Style loss** and decreasing these losses we would reach the perfect target image that has the style of one image and content of other image. For more learning checkout the links below.
 
  **Note:** The notebook has been ran on Google Colab, If you are working on local machine some starting four cells can be ignored. To use this and produce new styled images, just change the links to the **style** and **content** variables, change the path in the last cell, and include the path where you want to save the styled image and run the entire notebook. 
  

  

## Languages or Frameworks Used 

  * Python: language
  * NumPy: library for numerical calculations
  * Matplotlib: library for data visualisation
  * Pytorch: a deep learning framework by Facebook AI Research Team for building neural networks
  * torchvision: package consists of popular datasets, model architectures, and common image transformations for computer vision
  
## Setup
  
  To use this project, clone the repo
  
  ### Clone
  ```
    git clone https://github.com/Surya-Prakash-Reddy/Image-Style-Transfer-Pytorch.git
  ```
  
  After cloning, you can use the `Style Transfer.ipynb` notebook to learn or modify. If you want to use this and produce new styled images, just change the links to the **style** and **content** variables, change the path in the last cell, and include the path where you want to save the styled image and run the entire notebook.
  
## More Learning

* https://www.youtube.com/watch?v=R39tWYYKNcI&index=37&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF
* https://towardsdatascience.com/artistic-style-transfer-b7566a216431
