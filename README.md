# Intro to Neural Networks - WORKSHOP

# If you don't have Python:
### For OSx users:
* ```$ brew install python 2.7```

### For Interns
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
