# HandWriting-Recognition-Algorithm
This program uses Neutral Network Machine Learning algorithm to recognize hand writings from number 0 to 10. There are 5000 training examples in pixeldata.mat, where each training example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is represented by a foating point number indicating the grayscale intensity at that location.

The 20 by 20 grid of pixels is "unrolled" into a 400-dimensional vector. Each of these training examples becomes a single row in our data matrix X. This gives us a 5000 by 400 matrix X where every row is a training example for a handwritten digit image. The second part of the training set is a 5000-dimensional vector y that contains labels for the training set. To make things more compatible with MATLAB indexing, where there is no zero index, we have mapped the digit zero to the value ten. Therefore, a "0" digit is labeled as "10", while the digits "1" to "9" are labeled as "1" to "9" in their natural order.

The NNmain.m file is the main algorithm and all the others are functions.

![handwriting](https://cloud.githubusercontent.com/assets/10996578/8815694/ad6945b6-2fea-11e5-9cea-d74b3aa62172.jpg)
