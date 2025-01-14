---
layout: single
title:  "[딥러닝기초] 05. Classification 연습 (feat. Fashion Mnist)"
categories: DL
tag: [python, deep learning, pytorch, classification]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


# Lab 5 Multi-class Classification on FashionMNIST

You will 

- start doing image processing

- run our models on GPUs

- use the `DataLoader()` class to iterate over dataset

- visualise images in the dataset

- train the model in batches

- print training and test accuracies

- perform some transformations on the dataset

- tweak the hyper parameters to help with model performance



**IMPORTANT: Change your runtime type to GPU before continuining.**






```python
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import torch.optim as optim
import torch.nn as nn
```

### 0. Set up the device and hyper parameters



```python
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device) # should print cuda, change your runtime type to GPU if not

# Hyper parameters
input_size = 784 # 28x28
hidden_size = 32 # can be removed if you prefer to specify this in the model's class itself
num_classes = 10
num_epochs = 5 #5
batch_size = 32
learning_rate = 0.001
```

<pre>
cuda
</pre>
### 1. Load the dataset



```python
# import data
train_set = torchvision.datasets.FashionMNIST(root="./", download=True, 
                                              train=True,
                                              transform=transforms.Compose([transforms.ToTensor()]))

test_set = torchvision.datasets.FashionMNIST(root="./", download=False, 
                                              train=False,
                                              transform=transforms.Compose([transforms.ToTensor()]))

# TODO: after downloading, locate the data within your hosted runtime machine in the folder on the left
# Q0. Type 'OK' below if you see a folder called 'FashionMNIST'
# OK!!
```

<pre>
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ./FashionMNIST/raw/train-images-idx3-ubyte.gz
</pre>
<pre>
  0%|          | 0/26421880 [00:00<?, ?it/s]
</pre>
<pre>
Extracting ./FashionMNIST/raw/train-images-idx3-ubyte.gz to ./FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ./FashionMNIST/raw/train-labels-idx1-ubyte.gz
</pre>
<pre>
  0%|          | 0/29515 [00:00<?, ?it/s]
</pre>
<pre>
Extracting ./FashionMNIST/raw/train-labels-idx1-ubyte.gz to ./FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ./FashionMNIST/raw/t10k-images-idx3-ubyte.gz
</pre>
<pre>
  0%|          | 0/4422102 [00:00<?, ?it/s]
</pre>
<pre>
Extracting ./FashionMNIST/raw/t10k-images-idx3-ubyte.gz to ./FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ./FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
</pre>
<pre>
  0%|          | 0/5148 [00:00<?, ?it/s]
</pre>
<pre>
Extracting ./FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to ./FashionMNIST/raw

</pre>
<pre>
/usr/local/lib/python3.7/dist-packages/torchvision/datasets/mnist.py:498: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  /pytorch/torch/csrc/utils/tensor_numpy.cpp:180.)
  return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)
</pre>
### View basic info on training data



```python
print(f'Train set size: {len(train_set)}') # number of samples in training set 60000
print(f'Labels: {train_set.targets}') # displays targets labels for the training data 10
print(f'Count of each class: {train_set.targets.bincount()}') 

# Q1. Considering the size of the training set and the batch size, how many iterations will there be in each epoch?
  ## ANSWER: 60000/32 = 1875

# Q2. Is the dataset balanced? Why?
  ## ANSWER: The dataset seems perfectly balanced as each classes get same amount.
```

<pre>
Train set size: 60000
Labels: tensor([9, 0, 0,  ..., 3, 0, 5])
Count of each class: tensor([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000])
</pre>
### 2. Use DataLoader to load the datasets

If you get an error when you (re-)run the next code cell, then `restart the runtime` and run all code cells again.



```python
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

# Access the first data sample in the train_set using next(iter())
sample = next(iter(train_set))
print(f'Length: {len(sample)}')
print(f'Type: {type(sample)}')

# This means the data contains image-label pairs
# Unpack them
image, label = sample
# Same as these two lines:
# image = sample[0]
# label = sample[1]


print(image.shape)
print(label)

# Q3. What does the shape of the image tell you about the number of channels and dimensions of the images?
  ## ANSWER : The number of channel is 1 and the dimensions are 28*28.

# Q4. What does the label value represent?
  ## ANSWER : label value represents what kind of fashion item it is. That can be T-shirt / Trouser / Pullover / Dress / Coat / Sandal / Shirt / Sneaker / Bag / Ankel boot
```

<pre>
Length: 2
Type: <class 'tuple'>
torch.Size([1, 28, 28])
9
</pre>
### 3. Visualisation

View the first image



```python
plt.imshow(image.squeeze(), cmap='gray')
```

<pre>
<matplotlib.image.AxesImage at 0x7f2caba5ab10>
</pre>
<img src="data:image/png;base64,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"/>


```python
# Get the first BATCH from train_loader
batch = next(iter(train_loader))
print(len(batch))
print(type(batch))

# Unpack the images and labels
images, labels = batch

print(f'Image shape: {images.shape}')
print(f'Label shape: {labels.shape}')

# Q5. What does each number in the shape of the images represent?
  ## ANSWER : 32 - there are 32 images in the batch / 1 - the number of channel is 1 / 28, 28 : an image consists of 28*28 pixels

# Q6. What about the shape of the labels?
  ## ANSWER : There are 32 labels for each image.
```

<pre>
2
<class 'list'>
Image shape: torch.Size([32, 1, 28, 28])
Label shape: torch.Size([32])
</pre>
### View some sample images

- The table for the label index and description is available [HERE](https://github.com/zalandoresearch/fashion-mnist#labels)



```python
# Create a grid 
plt.figure(figsize=(15,15))
grid = torchvision.utils.make_grid(tensor=images, nrow=4) # nrow = number of images displayed in each row

print(f"class labels: {labels}")

# Use grid.permute() to transpose the grid so that the axes meet the specifications required by 
# plt.imshow(), which are [height, width, channels]. PyTorch dimensions are [channels, height, width].
plt.imshow(grid.permute(1,2,0), cmap='gray')


# Note that the images are grayscale (black and white) and have 28x28 pixels
# Grayscale images only have one channel
# TODO: Check that the image labels for each image corresponds to the correct label provided above

# Q7. How many images are displayed in total here and why? 
  ## ANSWER : 32 images / As the batch size is 32

# Q8. How do you increase or decrease the TOTAL number of images displayed?
  ## ANSWER : It is possible by adjusting the batch size
```

<pre>
class labels: tensor([9, 0, 0, 3, 0, 2, 7, 2, 5, 5, 0, 9, 5, 5, 7, 9, 1, 0, 6, 4, 3, 1, 4, 8,
        4, 3, 0, 2, 4, 4, 5, 3])
</pre>
<pre>
<matplotlib.image.AxesImage at 0x7f2cab504f50>
</pre>
<img src="data:image/png;base64,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"/>

### 4. Define the Neural Network, Loss and Optimiser



```python
# 4. NN model
class FashionNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super().__init__()
    self.layer1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.layer2 = nn.Linear(hidden_size, num_classes)
    # NOTE: softmax not added here because of CrossEntropyLoss later

  def forward(self, x):
    out = self.layer1(x)
    out = self.relu(out)
    out = self.layer2(out)
    return out

# 4.1 Create NN model instance
model = FashionNN(input_size, hidden_size, num_classes).to(device)

# 4.2 Loss and Optimiser
criterion = nn.CrossEntropyLoss() # will apply softmax
opt = optim.Adam(model.parameters(), lr=learning_rate)
```

### 5. Training Loop



```python
# 5. Training loop
n_total_steps = len(train_set)
n_iterations = -(-n_total_steps // batch_size) # ceiling division

n_correct = 0
n_samples = 0

for epoch in range(num_epochs):
  print('\n')
  # 5.1 loop over all the batches, i is index, (images, labels) is data
  for i, (images, labels) in enumerate(train_loader):
    # 5.2 Reshape images first [batch_size, 1, 28, 28] --> [batch_size, 784]
    # 5.3 Push images to GPU
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)

    # 5.4 Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # 5.5 Backward pass
    opt.zero_grad() # 5.6 Empty the values in the gradient attribute, or model.zero_grad()
    loss.backward() # 5.7 Backprop
    opt.step() # 5.8 Update params

    # 5.9 Print loss
    if (i+1) % 200 == 0:
      print(f'Epoch {epoch+1}/{num_epochs}, Iteration {i+1}/{n_iterations}, Loss={loss.item():.4f} ')

    # 5.10 Get Accuracy
    # torch.max() returns actual probability value (ignored) and index or class label (selected)
    _, y_preds = torch.max(outputs, 1)
    n_samples += labels.shape[0]
    n_correct += (y_preds == labels).sum().item()

# 5.11 Print accuracy
acc = 100.0 * n_correct / n_samples
print(f'\nTrain Accuracy = {acc:.4f}')
```

<pre>


Epoch 1/5, Iteration 200/1875, Loss=0.7098 
Epoch 1/5, Iteration 400/1875, Loss=0.5279 
Epoch 1/5, Iteration 600/1875, Loss=0.4319 
Epoch 1/5, Iteration 800/1875, Loss=0.6901 
Epoch 1/5, Iteration 1000/1875, Loss=0.3268 
Epoch 1/5, Iteration 1200/1875, Loss=0.3100 
Epoch 1/5, Iteration 1400/1875, Loss=0.8092 
Epoch 1/5, Iteration 1600/1875, Loss=0.4521 
Epoch 1/5, Iteration 1800/1875, Loss=0.4144 


Epoch 2/5, Iteration 200/1875, Loss=0.4737 
Epoch 2/5, Iteration 400/1875, Loss=0.3923 
Epoch 2/5, Iteration 600/1875, Loss=0.3724 
Epoch 2/5, Iteration 800/1875, Loss=0.5584 
Epoch 2/5, Iteration 1000/1875, Loss=0.2575 
Epoch 2/5, Iteration 1200/1875, Loss=0.2017 
Epoch 2/5, Iteration 1400/1875, Loss=0.7347 
Epoch 2/5, Iteration 1600/1875, Loss=0.3205 
Epoch 2/5, Iteration 1800/1875, Loss=0.4016 


Epoch 3/5, Iteration 200/1875, Loss=0.3962 
Epoch 3/5, Iteration 400/1875, Loss=0.3396 
Epoch 3/5, Iteration 600/1875, Loss=0.3641 
Epoch 3/5, Iteration 800/1875, Loss=0.5620 
Epoch 3/5, Iteration 1000/1875, Loss=0.2359 
Epoch 3/5, Iteration 1200/1875, Loss=0.1652 
Epoch 3/5, Iteration 1400/1875, Loss=0.6956 
Epoch 3/5, Iteration 1600/1875, Loss=0.2892 
Epoch 3/5, Iteration 1800/1875, Loss=0.3886 


Epoch 4/5, Iteration 200/1875, Loss=0.3749 
Epoch 4/5, Iteration 400/1875, Loss=0.3231 
Epoch 4/5, Iteration 600/1875, Loss=0.3466 
Epoch 4/5, Iteration 800/1875, Loss=0.5553 
Epoch 4/5, Iteration 1000/1875, Loss=0.2250 
Epoch 4/5, Iteration 1200/1875, Loss=0.1660 
Epoch 4/5, Iteration 1400/1875, Loss=0.6871 
Epoch 4/5, Iteration 1600/1875, Loss=0.3031 
Epoch 4/5, Iteration 1800/1875, Loss=0.3617 


Epoch 5/5, Iteration 200/1875, Loss=0.3490 
Epoch 5/5, Iteration 400/1875, Loss=0.3110 
Epoch 5/5, Iteration 600/1875, Loss=0.3305 
Epoch 5/5, Iteration 800/1875, Loss=0.5662 
Epoch 5/5, Iteration 1000/1875, Loss=0.2075 
Epoch 5/5, Iteration 1200/1875, Loss=0.1504 
Epoch 5/5, Iteration 1400/1875, Loss=0.6482 
Epoch 5/5, Iteration 1600/1875, Loss=0.3097 
Epoch 5/5, Iteration 1800/1875, Loss=0.3403 

Train Accuracy = 85.1877
</pre>
### 6. Evaluation



```python
# 6. Deactivate the autograd engine to reduce memory usage and speed up computations (backprop disabled).
with torch.no_grad():
  n_correct = 0
  n_samples = 0

  # 6.1 Loop through test set
  for images, labels in test_loader:
    # 6.2 0
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)
    # 6.3 Run on trained model
    outputs = model(images) 

    # 6.4. Get predictions
    # torch.max() returns actual probability value (ignored) and index or class label (selected)
    _, y_preds = torch.max(outputs, 1)
    n_samples += labels.shape[0]
    n_correct += (y_preds == labels).sum().item()

  # 6.5 Print accuracy
  acc = 100.0 * n_correct / n_samples
  print(f'Test Accuracy = {acc:.4f}')
```

<pre>
Test Accuracy = 86.0800
</pre>

```python
# Q9. What is the final loss of this model on the training set?
  ## ANSWER : 0.3018

# Q10. What is the accuracy of this model on the training set?
  ## ANSWER : 82.6842

# Q11. What is the accuracy of the trained model on the test set?
  ## ANSWER : 83.7300
```

### TASK: Increase `num_epochs` & redo

- Keep track of the loss and test set accuracy

- Set `num_epochs=5` in code cell 0 (hyper parameters)

- Rebuild and retrain the model by **running code cells 0, 4, and 5 ONLY**

- Evaluate the model on the test data by running **code cell 6**

- Answer the questions below



```python
# Q12. What is the final loss now and is it less than the previous loss?
  ## ANSWER: 0.3403 / It is more than the previous loss

# Q13. Are the training and test set accuracies higher now?
  ## YES / accuracies are both higher than the previous one

# Q14. After changing the num_epochs, why should code cell 4 (NN, Loss, Optimiser) be run before code cell 5 (training)?
  ## Because the "model" contains parameters and results of the previous model. So we have to refresh it in order to make another model.
```

## 6. Let's add some improvements



We are going to add ONE improvement at a time

- First the training data is **normalised and shuffled** (code provided). Use the same number of epochs as the the previous case to make a fair comparison.

- Build and train the model and get the loss, train and test set accuracies.

- Then change ONE hyper parameter, e.g. `num_epochs`, `hidden_size`, `batch_size`, `learning_rate` OR add layers.

- Run code cells 7-11 to train and test the model and take note of its loss, train and test accuracies.



**IMPORTANT!** If you have trouble running any of the code cells below, restart the runtime, via `Runtime-->Restart runtime` before continuing (or `Ctrl/Cmd + M + .`)




```python
# You DO NOT have to run this cell code unless you restarted the runtime.

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import torch.optim as optim
import torch.nn as nn
```

### 7. Normalise and Shuffle the Traning Data



```python
# Add Normalisation to transform data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.0,), (0.5,),)])

train_set = torchvision.datasets.FashionMNIST(root="./", download=True, 
                                              train=True,
                                              transform=transform)
test_set = torchvision.datasets.FashionMNIST(root="./", download=True, 
                                              train=False,
                                              transform=transform)
```

### 8a. Hyper-parameters



```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyper parameters
input_size = 784 # 28x28
hidden_size = 128
num_classes = 10 
num_epochs = 10
batch_size = 32
learning_rate = 0.001
```

<pre>
cuda
</pre>
### 8b. Shuffle the training data



This reshuffles the data at every epoch




```python
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
```

### 9. NN Model, Loss, Optimiser



```python
# 9. NN model
class FashionNN2(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super().__init__()
    self.layer1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.layer2 = nn.Linear(hidden_size, 32)
    self.layer3 = nn.Linear(32, num_classes)
    # NOTE: softmax not added here because of CrossEntropyLoss later

  def forward(self, x):
    out = self.layer1(x)
    out = self.relu(out)
    out = self.layer2(out)
    out = self.relu(out)
    out = self.layer3(out)
    return out

# 9.1 Create NN model instance
model = FashionNN2(input_size, hidden_size, num_classes).to(device)

# 9.2 Loss and Optimiser
criterion = nn.CrossEntropyLoss() # will apply softmax
opt = optim.Adam(model.parameters(), lr=learning_rate)
```

### 10. Training Loop



```python
# 10. Training loop
n_total_steps = len(train_set)
n_iterations = -(-n_total_steps // batch_size) # ceiling division

n_correct = 0
n_samples = 0

for epoch in range(num_epochs):
  print('\n')
  # 10.1 loop over all the batches, i is index, (images, labels) is data
  for i, (images, labels) in enumerate(train_loader):
    # 10.2 Reshape images first [batch_size, 1, 28, 28] --> [batch_size, 784]
    # 10.3 Push images to GPU
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)

    # 10.4 Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # 10.5 Backward pass
    opt.zero_grad() 
    loss.backward() 
    opt.step() 

    # 10.6 Print loss
    if (i+1) % 200 == 0:
      print(f'Epoch {epoch+1}/{num_epochs}, Iteration {i+1}/{n_iterations}, Loss={loss.item():.4f} ')

    # 10.7 Get model Accuracy
    # torch.max() returns actual probability value (ignored) and index of class label (selected)
    _, y_preds = torch.max(outputs, 1)
    n_samples += labels.shape[0]
    n_correct += (y_preds == labels).sum().item()

# 10.8 Print accuracy
acc = 100.0 * n_correct / n_samples
print(f'Finished training \nTrain Accuracy = {acc:.4f}')
```

<pre>


Epoch 1/10, Iteration 200/1875, Loss=0.7492 
Epoch 1/10, Iteration 400/1875, Loss=0.5174 
Epoch 1/10, Iteration 600/1875, Loss=0.5240 
Epoch 1/10, Iteration 800/1875, Loss=0.3046 
Epoch 1/10, Iteration 1000/1875, Loss=0.5334 
Epoch 1/10, Iteration 1200/1875, Loss=0.3833 
Epoch 1/10, Iteration 1400/1875, Loss=0.6227 
Epoch 1/10, Iteration 1600/1875, Loss=0.6400 
Epoch 1/10, Iteration 1800/1875, Loss=0.3058 


Epoch 2/10, Iteration 200/1875, Loss=0.4198 
Epoch 2/10, Iteration 400/1875, Loss=0.2854 
Epoch 2/10, Iteration 600/1875, Loss=0.4448 
Epoch 2/10, Iteration 800/1875, Loss=0.4668 
Epoch 2/10, Iteration 1000/1875, Loss=0.3642 
Epoch 2/10, Iteration 1200/1875, Loss=0.2993 
Epoch 2/10, Iteration 1400/1875, Loss=0.6645 
Epoch 2/10, Iteration 1600/1875, Loss=0.4463 
Epoch 2/10, Iteration 1800/1875, Loss=0.2599 


Epoch 3/10, Iteration 200/1875, Loss=0.2252 
Epoch 3/10, Iteration 400/1875, Loss=0.4819 
Epoch 3/10, Iteration 600/1875, Loss=0.2255 
Epoch 3/10, Iteration 800/1875, Loss=0.4261 
Epoch 3/10, Iteration 1000/1875, Loss=0.1820 
Epoch 3/10, Iteration 1200/1875, Loss=0.3556 
Epoch 3/10, Iteration 1400/1875, Loss=0.1780 
Epoch 3/10, Iteration 1600/1875, Loss=0.5367 
Epoch 3/10, Iteration 1800/1875, Loss=0.3226 


Epoch 4/10, Iteration 200/1875, Loss=0.1994 
Epoch 4/10, Iteration 400/1875, Loss=0.1792 
Epoch 4/10, Iteration 600/1875, Loss=0.3470 
Epoch 4/10, Iteration 800/1875, Loss=0.4402 
Epoch 4/10, Iteration 1000/1875, Loss=0.4358 
Epoch 4/10, Iteration 1200/1875, Loss=0.6902 
Epoch 4/10, Iteration 1400/1875, Loss=0.3412 
Epoch 4/10, Iteration 1600/1875, Loss=0.1993 
Epoch 4/10, Iteration 1800/1875, Loss=0.5635 


Epoch 5/10, Iteration 200/1875, Loss=0.1973 
Epoch 5/10, Iteration 400/1875, Loss=0.4644 
Epoch 5/10, Iteration 600/1875, Loss=0.2280 
Epoch 5/10, Iteration 800/1875, Loss=0.1326 
Epoch 5/10, Iteration 1000/1875, Loss=0.2262 
Epoch 5/10, Iteration 1200/1875, Loss=0.2439 
Epoch 5/10, Iteration 1400/1875, Loss=0.3043 
Epoch 5/10, Iteration 1600/1875, Loss=0.1140 
Epoch 5/10, Iteration 1800/1875, Loss=0.5503 


Epoch 6/10, Iteration 200/1875, Loss=0.2201 
Epoch 6/10, Iteration 400/1875, Loss=0.1644 
Epoch 6/10, Iteration 600/1875, Loss=0.1757 
Epoch 6/10, Iteration 800/1875, Loss=0.3857 
Epoch 6/10, Iteration 1000/1875, Loss=0.3420 
Epoch 6/10, Iteration 1200/1875, Loss=0.2592 
Epoch 6/10, Iteration 1400/1875, Loss=0.2144 
Epoch 6/10, Iteration 1600/1875, Loss=0.1722 
Epoch 6/10, Iteration 1800/1875, Loss=0.2085 


Epoch 7/10, Iteration 200/1875, Loss=0.1651 
Epoch 7/10, Iteration 400/1875, Loss=0.4621 
Epoch 7/10, Iteration 600/1875, Loss=0.2811 
Epoch 7/10, Iteration 800/1875, Loss=0.2457 
Epoch 7/10, Iteration 1000/1875, Loss=0.2578 
Epoch 7/10, Iteration 1200/1875, Loss=0.0426 
Epoch 7/10, Iteration 1400/1875, Loss=0.3199 
Epoch 7/10, Iteration 1600/1875, Loss=0.2743 
Epoch 7/10, Iteration 1800/1875, Loss=0.1175 


Epoch 8/10, Iteration 200/1875, Loss=0.1516 
Epoch 8/10, Iteration 400/1875, Loss=0.4050 
Epoch 8/10, Iteration 600/1875, Loss=0.2500 
Epoch 8/10, Iteration 800/1875, Loss=0.2755 
Epoch 8/10, Iteration 1000/1875, Loss=0.1106 
Epoch 8/10, Iteration 1200/1875, Loss=0.0927 
Epoch 8/10, Iteration 1400/1875, Loss=0.3019 
Epoch 8/10, Iteration 1600/1875, Loss=0.2352 
Epoch 8/10, Iteration 1800/1875, Loss=0.2726 


Epoch 9/10, Iteration 200/1875, Loss=0.2668 
Epoch 9/10, Iteration 400/1875, Loss=0.2192 
Epoch 9/10, Iteration 600/1875, Loss=0.1953 
Epoch 9/10, Iteration 800/1875, Loss=0.2328 
Epoch 9/10, Iteration 1000/1875, Loss=0.2469 
Epoch 9/10, Iteration 1200/1875, Loss=0.3755 
Epoch 9/10, Iteration 1400/1875, Loss=0.1565 
Epoch 9/10, Iteration 1600/1875, Loss=0.2965 
Epoch 9/10, Iteration 1800/1875, Loss=0.6138 


Epoch 10/10, Iteration 200/1875, Loss=0.2478 
Epoch 10/10, Iteration 400/1875, Loss=0.3233 
Epoch 10/10, Iteration 600/1875, Loss=0.0687 
Epoch 10/10, Iteration 800/1875, Loss=0.5231 
Epoch 10/10, Iteration 1000/1875, Loss=0.1747 
Epoch 10/10, Iteration 1200/1875, Loss=0.4084 
Epoch 10/10, Iteration 1400/1875, Loss=0.1478 
Epoch 10/10, Iteration 1600/1875, Loss=0.4027 
Epoch 10/10, Iteration 1800/1875, Loss=0.2333 
Finished training 
Train Accuracy = 88.1963
</pre>
### 11. Evaluation on Test Set



```python
# 11. Deactivate the autograd engine to reduce memory usage and speed up computations (backprop disabled).
with torch.no_grad():
  n_correct = 0
  n_samples = 0

  # 11.1 Loop through test set
  for images, labels in test_loader:
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)
    outputs = model(images) 

    # 11.2 Get predictions
    # torch.max() returns actual probability value (ignored) and index or class label (selected)
    _, y_preds = torch.max(outputs, 1)
    n_samples += labels.shape[0]
    n_correct += (y_preds == labels).sum().item()

  # 11.3 Print accuracy
  acc = 100.0 * n_correct / n_samples
  print(f'Test Accuracy = {acc:.4f}')
```

<pre>
Test Accuracy = 88.1100
</pre>

```python
# Q15. What is the final loss of this model on the training set?
  ## ANSWER : 0.2197

# Q16. What is the accuracy of this model on the training set?
  ## ANSWER : 83.3725

# Q17. What is the accuracy of the trained model on the test set?
  ## ANSWER : 84.6700

# Q18. Did shuffling and normalisation help to build a better model?
  ## ANSWER : The loss is decreased, and the accuracies of train/test were higher than the previous one(All hyperparameters were same but not shuffled and not normalized) not doing so.

# Q19. How many training samples were seen by the model during each epoch of training?
  ## ANSWER : 9 training samples were seen
```

### Further Improvement



- Try changing other hyper parameters, ONE at a time while keeping everything else the same

- **KEEP shuffling and normalisation**, i.e. do not change code cells 7 and 8b.

- Example changes:

  - Increase `num_epochs` (max 20)

  - Change number neurons in hidden layers (keep it within 128 max per layer)

  - Add ONE extra hidden layer at a time (start with 32 units). You only need to run code cells 9-11.

  - Change `batch_size` (to powers of 2s), try 64.

  - Change loss to NLLLoss(), you need to add softmax activation in the output layer

  - Change the learning rate



- Rerun **code cells 8a-11**(except when adding layers, where you would rerun just cells 9-11).

- If the change improves the model, KEEP this improvement and change something else and redo to see if it can be further improved.

- Keep a record of the final loss, train and test set accuracies for each test run.

- Make at LEAST THREE changes that help improve the model's performance on the test set.




```python
# Test 1 
# Change (e.g. num_epochs=10): num_epochs=10
# Loss: 0.1843
# Train acc: 89.7415
# Test acc: 86.6600
```


```python
# Test 2 
# Change (e.g. batch_size=64): Added one hidden layer
# Loss: 0.4372
# Train acc: 87.2048
# Test acc: 87.0200
```


```python
# Test 3
# Change: hidden_size=128
# Loss: 0.2333
# Train acc: 88.1963
# Test acc: 88.1100
```


```python
# Q20. State any insights you gained from this exercise.
  ## The loss and the accuracy has low correlation.
  ## Making a complex model can helpful for training performance, however that does not guarantee the test performance.
  ## Loss and accuracy varies for each runs. They are not always come to the similar results.
  ## Normalization can help increase the performance.
```
