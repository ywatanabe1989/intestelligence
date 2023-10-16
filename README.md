# Intestelligence: A Pharmacological Neural Network Using Intestine Data
The source code accompanying the archive article ["Intestelligence: A pharmacological neural network using intestine data" (https://www.biorxiv.org/content/10.1101/2023.04.15.537044v1)](https://www.biorxiv.org/content/10.1101/2023.04.15.537044v1).

#### Description
This repository contains the implementation of PerceptronOrINN, a three-layer neural network model designed to operate both as a Perceptron and as an Intestine-based Neural Network (INN). The model can be used for various classification tasks and has been specifically optimized for pharmacological applications.

#### Features
- Customizable activation functions
- Supports both Perceptron and INN modes
- Resampling activation functions based on given parameters
- Demonstrated on MNIST dataset

#### Installation and Usage
``` bash
# Clone the repository
$ git clone git@github.com:ywatanabe1989/intestelligence.git

# Navigate to the directory
$ cd intestelligence

# Create a Python virtual environment
$ python -m venv env

# Activate the virtual environment
$ source env/bin/activate

# Install the required packages
$ pip install ipython numpy torch torchvision

# Run the MNIST training script
$ python train_MNIST.py
```

#### File Structure
PerceptronOrINN.py: Contains the implementation of the PerceptronOrINN model.
train_MNIST.py: Script for training the PerceptronOrINN model on the MNIST dataset.

#### Dependencies
Python 3.x
NumPy
PyTorch
torchvision

#### Additional Notes
To switch between Perceptron and INN modes, you can update the act_str parameter in the model_config dictionary within the train_MNIST.py script.
