<h1>Image Style Transfer using PyTorch</h1>
<h2>About Project:</h2>
<p>
  <img src='https://github.com/SurajChinna/Image-Style-Transfer-Pytorch/blob/master/assets/download.png' />
  The main goal of project is to combine two images and produce new image. The combination works in slightly different way i.e., we         combine the style of one image with the content of other image. First we take the image from which we want to extract content usually     called <b>content image</b> and take another image from which the style is to be extracted usually called <b>style image</b>. This is     the implementation of <a href='https://arxiv.org/pdf/1508.06576.pdf'>this</a> research paper.  
</p>

<p>
  Convolutional Neural Networks are a type of neural networks which are used widely in Image classification and recongnition. A CNN         architecture called VGG19 has been used in this project. The starting layers in this architecture extract the basic features and shapes and later layers will extract more complex image patterns. So for the output image we will take the <b>content</b> from later layers of CNN. For extracting the style of image, we take the correlations between different layers using <a href="https://en.wikipedia.org/wiki/Gramian_matrix">Gram Matrix</a>
</p>

<p>
  Initially, we take any random image as target(or taking the content image would be useful) and compute the <b>Content loss</b> and <b>Style loss</b> and decreasing these losses we would reach the perfect target image that has the style of one image and content of other image. For more learning checkout the links below.
</p>

<h2>Languages or frameworks used</h2>
<p>
<ul>
  <li>Python: language</li>
  <li>NumPy: library for numerical calculations</li>
  <li>Matplotlib: library for data visualisation</li>
  <li>PIL: Python Image Library for opening and manage different image formats</li>
  <li>torch: a deep learning framework by Facebook AI Research Team</li>
  <li>torchvision: package consists of popular datasets, model architectures, and common image transformations for computer vision</li>
</ul>
</p>

<h2>More Learning</h2>
<p>
  <ul>
    <li>https://www.youtube.com/watch?v=R39tWYYKNcI&index=37&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF</li>
    <li>https://towardsdatascience.com/artistic-style-transfer-b7566a216431</li>
  </ul>
</p>
