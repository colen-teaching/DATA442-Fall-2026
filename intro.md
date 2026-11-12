# DATA 442 / 610 - Deep Learning and Neural Networks

This is the webpage for the Deep Learning and Neural Networks course held at William \& Mary in Fall 2026.

Classes will be held in ISC 2280 on Monday and Wednesday from 2:00 PM to 3:20 PM

Instructor Office Hours: Friday, 9:30-11:30 AM, ISC 3324

TA Office Hours: Tuesday, 12:00-1:00 PM, ISC 3364

## Course Description

This course teaches the foundation of Neural Networks and Deep Learning. Students entering into this course should have, at minimum, a background in data preprocessing, cleaning, manipulation, and dimensionality reduction within Python. Through an applied learning project, you will learn how to implement a machine learning project from design to implementation in the context of neural networks. Topics we will cover include the basic building blocks of neural networks, RNNs, convolutional networks and computer vision, backpropagation basics and strategies, and more. The course will utilize PyTorch.

## Syllabus

[Click here to download the Syllabus](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/files/DATA_442_Syllabus.pdf)

## Textbook

We will use the *Deep Learning* book by Goodfellow, Bengio, and Courville, which is available online for free at [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)

## Schedule

The course schedule will be updated regularly. Dates and topics may change as the semester progresses, so please check this regularly.  

| Week | Day | Lecture Topic  | In-class Demo |
| ---- | --- | -------------- | -------- |
| 1 | Aug 26 | [Course Introduction & History of Deep Learning](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/lectures/Lecture%201%20-%20Course%20Introduction.pdf)
| 2 | Aug 31 | [Linear Classifiers & Perceptrons](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/lectures/Lecture%202%20-%20Linear%20Classifiers.pdf) | [Perceptron](/binary-perceptron)  [Adaline](/adaline) [Convergence Proof](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/files/Perceptron_Convergence_Proof.pdf)|
|   | Sep 2 | [Stochastic Gradient Descent, Logistic Regression](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/lectures/Lecture%203%20-%20Logistic%20Regression.pdf) | [SGD](/stochastic-gradient-descent)|
| 3 | Sep 7 | No class - Labor Day Holiday |
|   | Sep 9 | [Logistic Regression, Regularization](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/lectures/Lecture%204%20-%20Regularization.pdf) | [LR](/logistic-regression-regularization)|
|   | Sep 11 | <span style="color:red">PS 0 Due (bonus): Getting started</span>| [link](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/files/DATA_442_PS0.pdf) |
| 4 | Sep 14 | [Optimization, Hyperparameter Tuning](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/lectures/Lecture%205%20-%20Optimization.pdf) | [Hyperparameter tuning](/hyperparameters)|
|   | Sep 16 | No class | 
| 5 | Sep 21 | [Deep Neural Networks & Backpropagation, from scratch](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/lectures/Lecture%206%20-%20Deep%20NNs.pdf) | [NNs from scratch](/deep-nns)|
|   | Sep 23 | [Deep Neural Networks & Backpropagation, in PyTorch](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/lectures/Lecture%207%20-%20PyTorch.pdf) | [PyTorch](/pytorch) |
| 6 | Sep 28 | [PyTorch & AutoGrad](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/lectures/Lecture%208%20-%20Autograd.pdf) | [PyTorch NNs](/pytorch-nn) [Autograd](/pytorch-autograd) |
|   | Sep 30 | [Image Data, Convolutions](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/lectures/Lecture%209%20-%20Images.pdf) | [Convolutions](/images-cnns) |
|   | Oct 2 | <span style="color:red">PS 1 Due: Building a NN from scratch</span>| [link](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/files/DATA_442_PS1.pdf) |
| 7 | Oct 5 | [Convolutional Neural Networks](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/lectures/Lecture%2010%20-%20CNNs.pdf) | [CNN classification](/cnn-classification)
|   | Oct 7 | [CNN Topics](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/lectures/Lecture%2011%20-%20CNN%20Topics.pdf) | [CNN Regression](/cnn-regression) [Torchvision](/torchvision) |
| 8 | Oct 12 | [CNN Autoencoders](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/lectures/Lecture%2012%20-%20Encoder%20Decoder.pdf) | [Autoencoder](/encoder-decoder) [SLURM sample](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/files/example_script.slurm) | 
|   | Oct 14 | [Recurrent Neural Networks](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/lectures/Lecture%2013%20-%20RNNs.pdf) | [RNN Classification](/rnn-classification) |
| 9 | Oct 19 | [RNN variants](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/lectures/Lecture%2014%20-%20RNNs.pdf) | [RNN Classification](/rnn-classification) [RNN Forecasting](/rnn-forecasting)
|   | Oct 21 | [RNN forecasting](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/lectures/Lecture%2015%20-%20Time%20Series.pdf) | [RNN Forecasting](/rnn-forecasting) 
|   | Oct 25 | <span style="color:red">PS 2 Due: CNNs and Image Data</span>| [link](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/files/DATA_442_PS2.pdf) |
| 10 | Oct 26 | [Graphs, Graph Convolutions](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/lectures/Lecture%2016%20-%20Graph%20Convolutions.pdf) | [Graphs](/gnn-introduction)
|    | Oct 28 | [Graph Neural Networks](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/lectures/Lecture%2017%20-%20GNNs.pdf) | [GNNs with Torch-Geometric](/gnn-pytorch)
| 11 | Nov 2 | [Transformers](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/lectures/Lecture%2018%20-%20Transformers.pdf) | [Transformers](/transformers)
|    | Nov 4 | [Representation learning](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/lectures/Lecture%2019%20-%20Representations.pdf) | [Representations](/representations)
|   | Nov 8 | <span style="color:red">PS 3 Due: RNNs and GNNs</span>| [link](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/files/DATA_442_PS3.pdf) |
| 12 | Nov 9 | [Variational autoencoders](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/lectures/Lecture%2020%20-%20VAEs.pdf) | [VAE](/vae) |
|    | Nov 11 | [Generative adversarial networks](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/lectures/Lecture%2021%20-%20GANs.pdf) | [GAN](/gan) |
| 13 | Nov 16 | GAN variants | [DCGAN](/dcgan) |
|    | Nov 18 | Contrastive representation learning |
| 14 | Nov 23 | TBD: Discuss Final Project |
|    | Nov 25 | No class - Thanksgiving Holiday | <span style="color:red">PS 4 Due: VAEs [link](https://raw.githubusercontent.com/colen-teaching/DATA442-Fall-2026/refs/heads/main/files/DATA_442_PS4.pdf) </span>|
| 15 | Nov 30 | (Tentative) Interpretability, GradCAM |
|    | Dec 2 | (Tentative) Physics-inspired deep learning |
| Finals | Dec 7-15 | Final Project Report and Presentation |