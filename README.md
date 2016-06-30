# Intro to Neural Networks - WORKSHOP

[Slides] (https://slides.com/chunkzer/deck)

# If you don't have Python:
### For OSx users:
* ```$ brew install python 2.7```

### For Linux
* ```$ sudo add-apt-repository ppa:fkrull/deadsnakes```
* ```$ sudo apt-get update```
* ```$ sudo apt-get install python2.7```


# If you don't have pip:

* ```$ sudo easy_install pip```
* ```$ sudo easy_install --upgrade six``` 

#If you don't have Tensorflow:

1.0 Select the appropiate binary

``` 
Ubuntu/Linux 64-bit, CPU only, Python 2.7 :
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0rc0-cp27-none-linux_x86_64.whl
Mac OS X, CPU only, Python 2.7:
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0rc0-py2-none-any.whl 
```

2.0 Then use pip to install:

* ```$ sudo pip install --upgrade $TF_BINARY_URL```

3.0 You're all set up.

------------------------------------------------

# Additional resources:

## For Neural Networks:

* http://karpathy.github.io/neuralnets/  

-- Most pragmatic explanation of backpropagation and gradient descent I've seen.

* http://neuralnetworksanddeeplearning.com/

-- Michael Nielsens ongoing "book." Very detailed in its steps.

### For Convolutional Neural Networks

* http://colah.github.io/posts/2014-07-Understanding-Convolutions/


### For Reccurrent Neural Networks:

* http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 

-- Excellent explanation of RNN inner workings.

* http://karpathy.github.io/2015/05/21/rnn-effectiveness/

-- Excellent *demonstration* of RNN Power.

* https://github.com/karpathy/char-rnn

-- Original char-rnn python implementation. Done using Torch.

* https://openai.com/blog/generative-models/

-- Overview of the cutting edge on generative models. Their current state and possible future.

