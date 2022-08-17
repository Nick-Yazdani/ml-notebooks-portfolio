# Machine Learning & Deep Learning Portfolio

## 1. Diamond Valuation using CNN/DNN

### Introduction

This is a demonstration of some of my capacity as Machine Learning Engineer, 
I had previously reacched an excellent result but the code was poorly organised and 
in a notebook on Kaggle. This is an adaptation of how I would handle this problem if 
I was working in a professional capacity.

#### Motivation

The reason I like this dataset and its resultant code implementation is that it requires the developer to deviate from
commonly presented examples of TensorFlow applications there is a requirement to use both tabular and image data in conjunction
to train the model and make meaningful and realistic predictions. 

Training on either the images or the tabular data alone is much simpler, that is why I chose to work on this dataset,
because it demonstrates competency in the following areas of TensorFlow

- Basic Regression in TensorFlow with tabular data (CSV etc.)

- Convolutional Neural Networks in TensorFlow, which also imply command of the simpler classification architectures (binary classifer and multi-class) as the only difference
is that the logits, instead of being passed through a binary cross-entropy or softmax loss function for the purposes of GD (depending on how many classes there are of course), it is passed to an appropriate merge layer which essentially allows the network to learn from both types of inputs simultaneously when trainig

- How to use a Convolutional Neural Network to do something other than just build an image classification model

- Using the Functional API to define your model architecture, which is less intuitive but more flexible and ultimately required whenever you are dealing with Deep Learning problems where a simple regression or classification model that only takes in one form of data will not be sufficient or even useful (GANs are a good example where
the Functional API is a hard requirement)

- Merge layers (how do you take the ouptuts of different layers of a Neural Network, that are in totally different formats, to the point in this example
they are different file types even, on the low level the tabular data is represented as feature vectors which you would expect with access to
a data dictionary and even just meaningful column names, a human could derive informatoin from a row, and perhaps with a simple enough data set could even
make some hypotheses about patterns that could be useful in making better predictions whereas the image data is a far more complicated construct which has information that is not interpretable by humans at all directly, i.e. if you print the output of `plt.imread(image) to stdout without first looking at the picture, what have you now learned?

- This codebase also demonstrates other aspects required through the whole machine learning life cycle, even the data is initially spread across multiple CSV files,
it then has to be cleaned and wrangled into a form the model accepts, there is also a small amount of feature engineering

- Unrelated to ML but the image files are very poorly mapped initially between their dataframe entry, this requires adhoc Python code to rectify
   
##### Process / Method

1. Data is fragemented across several csv files so this step requires them to be loaded into memory then concatenated into a single datafraem
2. Data cleansing and feature engineering 
    - Split measurement column into three seperate columns so that they can be input as a numeric type into the model once all alpha/symbols are removed from each row
    - Remove all non-numeric characters
3. Perform some rudimentary image preprocessing and normalization
4. Use the TensorFlow datasets API to create the required train/val/split datasets (this is for ease of use and also efficiency)
5. Build and train modedel
6. Evaluate model performance
7. Adjust hyper-paramters or make any other relevant changes and then repeat 5

## 2. Dog vs Cat Binary Classifier CNN in TensorFlow

### Introduction

This is a basic adaptation of one of the TensorFlow examples on building an image classifier. In this example it has been used for classification of images that are either images of a dog or a cat.

#### Motivation

Just a straightforward example demonstrating my ability to leverage a basic CNN architecture for binary image classification using TensorFlow
