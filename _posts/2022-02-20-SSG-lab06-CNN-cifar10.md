---
layout: single
title:  "[딥러닝기초] 06. Convolution & CNN (feat. Cifar10)"
categories: DL
tag: [python, deep learning, pytorch, CNN]
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


## Lab 6 (Image Processing using Convolutional Neural Networks)

- CIFAR10 dataset (see https://www.cs.toronto.edu/~kriz/cifar.html for more info)

- 60K images: 50K train, 10K test

- 10 classes: 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'

- Perform multi-class classification with evaluation accuracy on EACH class



**CONNECT TO GPU** before continuing, but just CPU is also fine, it might be a bit slow.




```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyper parameters
num_epochs = 5
batch_size = 4
learning_rate = 0.001

# Download and prepare dataset
# Transform them to tensors and normalise them
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

# 2.2 Download data
train_set = torchvision.datasets.CIFAR10("./", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10("./", train=False, download=True, transform=transform)

# 2.3 Use DataLoader to get batches and shuffle
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Q1. Why are there 3 values in each list of the Normalize() function? What do they represent?
  ## The first chunk((0.5, 0.5, 0.5)) refers to each channel's means
  ## The second chunk((0.5, 0.5, 0.5)) refers to each channel's stadard diviations.
  ## By doing so, the value range has became [-1,1]
```

<pre>
cuda
Files already downloaded and verified
Files already downloaded and verified
</pre>
### Inspect the Images



```python
# Access the first data sample in the train_set using next(iter())
batch = next(iter(train_loader))
print(f'Image values: \n{batch}')
print(f'Length: {len(batch)}')
print(f'Type: {type(batch)}')

# This means the data contains image-label pairs
# Unpack them
images, labels = batch
# Same as these two lines:
# image = batch[0]
# label = batch[1]


print(images.shape)
print(labels)

# Q2. What is the range of the values for the normalised image pixels?
  ## [-1,1]

# Q3. What does each index value of the shape of the image represent?
  ## [4, 3, 32, 32] means the batch has 4 images, images are RGB scaled, an image consists of 32*32 pixels.

# Q4. What do the label values represent?
  ## label values [6, 4, 7, 3] refers to each image's class.
```

<pre>
Image values: 
[tensor([[[[-0.9529, -0.8588, -0.8667,  ..., -0.9922, -0.9686, -0.9686],
          [-0.9765, -0.9294, -0.9216,  ..., -0.9922, -0.9843, -0.9686],
          [-0.9843, -1.0000, -0.9922,  ..., -0.9922, -0.9922, -0.9686],
          ...,
          [ 0.1059,  0.4510,  0.4588,  ..., -0.8275, -0.6078, -0.4039],
          [-0.1608,  0.2314,  0.4902,  ..., -0.7490, -0.4980, -0.6078],
          [-0.1294, -0.1608,  0.3333,  ..., -0.2706, -0.2000, -0.4196]],

         [[-0.9451, -0.8588, -0.8588,  ..., -0.9922, -0.9686, -0.9686],
          [-0.9686, -0.9294, -0.9216,  ..., -0.9922, -0.9843, -0.9686],
          [-0.9843, -1.0000, -1.0000,  ..., -0.9922, -0.9922, -0.9686],
          ...,
          [ 0.1451,  0.5686,  0.6471,  ..., -0.8039, -0.5608, -0.3490],
          [-0.1059,  0.3255,  0.6392,  ..., -0.6941, -0.4353, -0.5373],
          [-0.1294, -0.1373,  0.4196,  ..., -0.1843, -0.1216, -0.3412]],
    
         [[-0.9843, -0.8902, -0.8745,  ..., -0.9922, -0.9686, -0.9686],
          [-0.9922, -0.9373, -0.9059,  ..., -0.9922, -0.9843, -0.9686],
          [-0.9922, -0.9843, -0.9608,  ..., -0.9922, -0.9922, -0.9686],
          ...,
          [-0.1843,  0.3412,  0.5373,  ..., -0.9686, -0.7490, -0.5451],
          [-0.3961,  0.0902,  0.5216,  ..., -0.8980, -0.6157, -0.6941],
          [-0.4353, -0.4275,  0.2549,  ..., -0.4353, -0.3569, -0.5216]]],


        [[[-0.4353, -0.4980, -0.4431,  ..., -0.3020, -0.2706, -0.3961],
          [-0.4510, -0.4745, -0.4588,  ..., -0.3647, -0.3490, -0.3725],
          [-0.3490, -0.4118, -0.3882,  ..., -0.3961, -0.4510, -0.4588],
          ...,
          [-0.5373, -0.6392, -0.6314,  ...,  0.3098,  0.1765,  0.3098],
          [-0.5059, -0.6471, -0.6157,  ...,  0.1922,  0.2235,  0.3020],
          [-0.5216, -0.6235, -0.4196,  ...,  0.2392,  0.2000,  0.3255]],
    
         [[-0.0353, -0.0902, -0.0196,  ...,  0.0824,  0.1137,  0.0118],
          [-0.0588, -0.0667, -0.0431,  ...,  0.0275,  0.0431,  0.0431],
          [ 0.0510, -0.0039,  0.0353,  ...,  0.0039, -0.0431, -0.0588],
          ...,
          [ 0.0275, -0.1137, -0.1294,  ...,  0.2235,  0.1608,  0.3255],
          [-0.0902, -0.2863, -0.2941,  ...,  0.0588,  0.1608,  0.2863],
          [-0.1686, -0.3176, -0.1765,  ...,  0.1686,  0.1608,  0.3098]],
    
         [[ 0.0039,  0.0039,  0.0588,  ...,  0.1216,  0.1451,  0.0353],
          [-0.0118,  0.0118,  0.0196,  ...,  0.0745,  0.0745,  0.0745],
          [ 0.0902,  0.0510,  0.0824,  ...,  0.0667, -0.0039, -0.0118],
          ...,
          [ 0.3176,  0.1451,  0.1137,  ..., -0.0667, -0.1294, -0.0118],
          [ 0.1373, -0.1137, -0.1451,  ..., -0.2157, -0.1137, -0.0118],
          [ 0.0275, -0.1686, -0.0745,  ..., -0.1216, -0.1216, -0.0118]]],


        [[[-0.2392, -0.2157, -0.2235,  ..., -0.4196, -0.4588, -0.4118],
          [-0.2549, -0.2235, -0.1843,  ..., -0.4196, -0.3804, -0.2784],
          [-0.2471, -0.0824, -0.1059,  ..., -0.2314, -0.1843, -0.0980],
          ...,
          [-0.0118,  0.0745,  0.1529,  ...,  0.7725,  0.7255,  0.7412],
          [ 0.2863,  0.1765,  0.0745,  ...,  0.6941,  0.7412,  0.7647],
          [ 0.3176,  0.3176,  0.2235,  ...,  0.7255,  0.7647,  0.7725]],
    
         [[-0.2863, -0.2157, -0.1765,  ..., -0.3569, -0.3961, -0.3725],
          [-0.3020, -0.2157, -0.1373,  ..., -0.3255, -0.2863, -0.2157],
          [-0.2863, -0.0745, -0.0588,  ..., -0.1137, -0.0667,  0.0039],
          ...,
          [-0.0353,  0.1059,  0.2392,  ...,  0.4980,  0.4588,  0.4745],
          [ 0.1137,  0.0745,  0.0510,  ...,  0.4118,  0.4510,  0.4745],
          [ 0.0824,  0.1216,  0.0745,  ...,  0.4431,  0.4745,  0.4824]],
    
         [[-0.5216, -0.5608, -0.5608,  ..., -0.6471, -0.6627, -0.6157],
          [-0.5373, -0.5686, -0.5216,  ..., -0.6863, -0.6392, -0.5529],
          [-0.5216, -0.4275, -0.4431,  ..., -0.5451, -0.5137, -0.4431],
          ...,
          [-0.0902,  0.0588,  0.2235,  ...,  0.3255,  0.2863,  0.2941],
          [ 0.0196, -0.0196, -0.0275,  ...,  0.2314,  0.2706,  0.2941],
          [-0.0510, -0.0039, -0.0510,  ...,  0.2627,  0.2941,  0.3020]]],


        [[[ 0.4039,  0.4275,  0.4510,  ...,  0.2941,  0.2314,  0.2157],
          [ 0.4588,  0.4667,  0.4824,  ...,  0.2863,  0.2157,  0.2235],
          [ 0.4824,  0.4824,  0.4980,  ...,  0.2863,  0.2314,  0.2157],
          ...,
          [ 0.3647,  0.4353,  0.5137,  ...,  0.7647,  0.6549,  0.4588],
          [ 0.4510,  0.4196,  0.4667,  ...,  0.6471,  0.5059,  0.2314],
          [ 0.4980,  0.5686,  0.5922,  ...,  0.4824,  0.3333, -0.0510]],
    
         [[ 0.4196,  0.4353,  0.4588,  ...,  0.3725,  0.3176,  0.2941],
          [ 0.4667,  0.4667,  0.4824,  ...,  0.3569,  0.3098,  0.3490],
          [ 0.4745,  0.4667,  0.4824,  ...,  0.3647,  0.3490,  0.3647],
          ...,
          [ 0.4039,  0.4588,  0.5216,  ...,  0.7412,  0.6392,  0.4510],
          [ 0.4902,  0.4510,  0.4745,  ...,  0.6471,  0.4902,  0.2157],
          [ 0.5451,  0.6000,  0.6157,  ...,  0.4824,  0.3255, -0.0588]],
    
         [[ 0.3961,  0.4196,  0.4431,  ...,  0.5529,  0.4980,  0.4745],
          [ 0.4353,  0.4353,  0.4510,  ...,  0.5529,  0.5059,  0.5373],
          [ 0.4275,  0.4275,  0.4353,  ...,  0.5373,  0.5294,  0.5529],
          ...,
          [ 0.3961,  0.4588,  0.5294,  ...,  0.7804,  0.7098,  0.5765],
          [ 0.5059,  0.4667,  0.4980,  ...,  0.7176,  0.6157,  0.3725],
          [ 0.5765,  0.6314,  0.6549,  ...,  0.6000,  0.4824,  0.0902]]]]), tensor([6, 4, 7, 3])]
Length: 2
Type: <class 'list'>
torch.Size([4, 3, 32, 32])
tensor([6, 4, 7, 3])
</pre>
### View some images

- Note that images have been normalised and may not look very clear



```python
# Create a grid 
plt.figure(figsize=(12,12))
grid = torchvision.utils.make_grid(tensor=images, nrow=4) # nrow = number of images displayed in each row

print(f"class labels: {labels}")

# Use grid.permute() to transpose the grid so that the axes meet the specifications required by 
# plt.imshow(), which are [height, width, channels]. PyTorch dimensions are [channels, height, width].
plt.imshow(grid.permute(1,2,0))
```

<pre>
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
</pre>
<pre>
class labels: tensor([6, 4, 7, 3])
</pre>
<pre>
<matplotlib.image.AxesImage at 0x7fa68d260310>
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAr8AAADPCAYAAAD8pLkGAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3df5xcdXkv8M8zmQyTYVg2m2VZkiUuYVljiDGmkRuBUoyIERW4Xq8i1kpri97Wi17t7aXa29aW22sr1eq12kbxikhFQURAQECggJCEEEIIISRhWcLmB8tmmWyWZRkm871/7HDNOZ8n7GF29kcyn/frxYucJ2dmvjPnx56dfD/nsRACRERERETqQWqyByAiIiIiMlF08SsiIiIidUMXvyIiIiJSN3TxKyIiIiJ1Qxe/IiIiIlI3dPErIiIiInVjTBe/ZrbCzJ40s21mdmmtBiUiIiIiMh6s2vv8mtk0AFsAvAtAD4CHAHwkhLDpNR6jmwqLiIiIyEToCyEcEy+O5ZvfUwBsCyF0hRCKAK4BcN4Ynk9EREREpFae8YpjufidA+DZA5Z7KjURERERkSkpPd4vYGYXA7h4vF9HRERERGQ0Y7n43QHg+AOW2yq1iBDCSgArAc35FREREZHJNZZpDw8BOMnMTjCzDIALANxYm2GJiIiIiNRe1d/8hhBKZvZpAL8EMA3A90IIj9dsZCIiIiIiNVb1rc6qejFNexARERGRifFwCGFpvKgObyIiIiJSN3TxKyIiIiJ1Y9xvdSZyeLAE6zQ4Dxvk2uwmrpXKXHtuT4LXBHD8rOjys87jjpvFtbLzmg3NXGtqdR7rvK/e/uhygVfB3iGnOOzUnFPTTOfzhfN8LzzvrJfENKeW49KMPNeGnc+jObadsxleZ2cP1/a/5I7uUHdy7C7wl/zxe2idyy67lWoF5+NoPY5rW3dx7f3vmE61wf5XIst3P8qPm8rmHM21HXsnfhyeT/7JH1Etk+b9PpOJ1tLO13AZp5h2niuV8tbjmrcereN9H5jic1GpyKuVSiXnNb1LLF4Pqfi5mMdRLnFtuMjP1TvA56L+AR5wz04+Qff3R8+naednRDrD58S2zoVUyzS28GOdbTA80E+1wsBAZLlU5PGv/sX/ptrroW9+RURERKRu6OJXREREROqGLn5FREREpG7o4ldERERE6oYCbyKJeLeojgeknPBV2M+13U4gq+mYagY1wgu4xe1y1jlhDteGnSTHti6uNWS51h8LWuxzngv7nJoXNHPCYVknjJdz1ovn514a4HXghP3gBNngPPYl57EznDBePHjiZf2yTqDuxUMs8HaEU3uZS/2xXEtnB4dk2ts58HbfE/xcnc7HPcMJvO3e+QrV1j7J600JnM0DePhIObsMpkjgrSHPg/O+YYsH0jJeaM2ppbwQnLNeNsvnp3R69MsdLxSXyfBzpdxLJyek5p5nuJZKlUZZAygVueqF7FqbeWxFJyzX39ZIta7unZHljZs4kFsocKCutci1fIrP1ykn2F128n+lcnS8qQYe61jpm18RERERqRu6+BURERGRuqGLXxERERGpG5rzK5LEUcdybV9sEucMZx5szjnEik5TB6+RxPPVNmtIaNCZWTbojM2b4wrnvbpzfGOOPdl5Tb7JOV50JnD27uZaozOOl+LP58y7ducZv+DUvIc6E07zzpy0vth4X074/IcaZ86e56zlx0eWi8P8wI553L1iwxO8L/Q4vUGyztzj7du51j4zuvz0VNkszvxebx5wewfXCs5hu6/a9+X18/EiD44MNWuAO4E1HdtpvDm0XsMJv/GFdxnjzKt11orPIS47TR3KzvnamwecZE4x4De+4Jozjowz59eZB4wSP7/Xz6gpx+ex1oZohmJuCzequH/NJqqte+BOqs1u76TaPGeev7dh0rEsRynhZ/t66JtfEREREakbuvgVERERkbqhi18RERERqRtjmkhhZt0YuXHnfgClEMLSWgxKRERERGQ81GIW8TtCCH01eJ4J87Y3cG3ZouOp9n9uenYCRiOHBOfm7RiOBd7KTuAr4zyukUMEGHAeO2Mm116qMsXi3UC/2RlbyhmH996zzj8aDcZuav6y81zDfDN0eDftTztNP7xglRfQcwNuSdbxQnDO4LyHDjnvNR1rwPHKkc4DvTBh9U0uZjkf2554c40Xq356n/N5zHJ23d5YALC7m1Nrw04IrmUWP1dfgWsFZxzeD7ick5GcspwQ3MaNXNvnfB5V847HhPtMJpUs/UhhNjcn54W5kjW+gNNMwRtZNhU9RstOxwUvLOZJp5wdyxvaMD9hOhagSzuhuJLz88VvBMKNf7zjytMQC5p1tvF7ams+nWq3P7CWajfdejPVHvv1Vc6r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"/>

## CNN model



```python
class Test(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5) 
    # flatten 3D tensor to 1D tensor
    self.fc1 = nn.Linear(400, 128) # Q8. Fill out the correct input dimensions  -> 16*5*5
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 10) # final output matches num_classes

  def forward(self, x):
    # Conv + ReLU + pool
    print(f'Input shape: {x.shape}')
    out = self.conv1(x)
    print(f'After Conv1: {out.shape}')
    print(f'Padding: {self.conv1.padding}')
    out = self.pool(F.relu(out))
    print(f'After Pool1: {out.shape}')
    out = self.conv2(out)
    print(f'After Conv2: {out.shape}')
    out = self.pool(F.relu(out))
    print(f'After Pool2: {out.shape}')
    # Flatten it before fc1
    out = out.reshape(-1, 400) # Q8. Fill out the correct dimension after -1
    print(f'Before fc1: {out.shape}')
    out = self.fc1(out)
    out = self.relu(out)
    print(f'After fc1: {out.shape}')
    out = self.fc2(out)
    out = self.relu(out)
    print(f'After fc2: {out.shape}')
    out = self.fc3(out) # NO softmax as it will be included in CrossEntropyLoss
    print(f'After fc3: {out.shape}')
    return out


model = Test().to(device)
# Let's view the softmax output
probs = nn.Softmax(dim=1)


# Q5. What do the three arguments of the first convolutional layer, conv1 represent (3,6,5)? 
  ## The number of input channel is 3 / # of output channel is 6 / the kernel size is 5*5

# Q6. Explain the arguments of the second convolutional layer, conv2 (6, 16, 5) 
  ## The number of output channel of the image that came through (conv1 -> pool1) was 6.
  ## So the input channel size is 6 / we want to make the output channel size as 16 / and the kernel size is still same (5*5)

# Q7. Figure out the convolved image size after conv1
# Convolved image size = ((input_width - filter_size + 2 * padding) / stride) + 1
  ## ((32-5+2*0) / 1) + 1 = 28
  ## So the image size is 28*28

# Q8. Figure out the input size to the first fcn layer and fill out the code above in init() and forward()
```

### Run through a sample batch



```python
sample = next(iter(train_loader))

images, labels = sample

images = images.to(device)
labels = labels.to(device)

output = model(images)
print(f'Output shape: {output.shape}')
print(f'Softmax outputs:\n {probs(output)}')


# Q9. Explain the shape of the output after conv1
  ## The batch has 4 images / number of channel is 6 / the image size is 28*28

# Q10. What does the pooling do to the dimensions of the feature images here?
  ## It reduces the image size into half.

# Q11. Add padding=1 to conv1 and rerun the last two code cells. How did padding affect the dimensions of the feature images?
  ## It made the reduction of image size slower.
  ## By the way, the output size after poo2 was same as before.

# Q12. What is represented by each list returned by Softmax outputs?
  ## The 10 values of each list means the probablity of which class the image belongs to.
  ## Within 10 values, CNN model gives class to an image using the highest value(probability).
```

<pre>
Input shape: torch.Size([4, 3, 32, 32])
After Conv1: torch.Size([4, 6, 30, 30])
Padding: (1, 1)
After Pool1: torch.Size([4, 6, 15, 15])
After Conv2: torch.Size([4, 16, 11, 11])
After Pool2: torch.Size([4, 16, 5, 5])
Before fc1: torch.Size([4, 400])
After fc1: torch.Size([4, 128])
After fc2: torch.Size([4, 64])
After fc3: torch.Size([4, 10])
Output shape: torch.Size([4, 10])
Softmax outputs:
 tensor([[0.0941, 0.1036, 0.1133, 0.0987, 0.1046, 0.0839, 0.0905, 0.0966, 0.1096,
         0.1051],
        [0.0920, 0.1038, 0.1138, 0.1012, 0.1045, 0.0844, 0.0911, 0.0945, 0.1087,
         0.1061],
        [0.0939, 0.1039, 0.1132, 0.0996, 0.1034, 0.0840, 0.0897, 0.0958, 0.1093,
         0.1072],
        [0.0938, 0.1038, 0.1134, 0.0998, 0.1055, 0.0836, 0.0901, 0.0951, 0.1092,
         0.1057]], device='cuda:0', grad_fn=<SoftmaxBackward>)
</pre>


### Let's Train!

- Now that we know and understand how CNNs work, let's put everything together for CIFAR-10 dataset

  - Download the data in batches and normalisation with shuffling

  - Build a model with 2 CNN layers containing ReLU and pooling

  - Passing the feature images to 3 fully connected layers (FCNs) also containing RELU activation

  - The final layer has 10 units to reprsent the number of output classes

  - Use Binary Cross Entropy Loss and SGD optimiser

  - Evaluate the model on the test data on EACH class



**IMPORTANT!** Fill out the missing code below before training 



```python
class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5) 
    # flatten 3D tensor to 1D tensor
    self.fc1 = nn.Linear(400, 256) # TODO
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 64)
    self.fc4 = nn.Linear(64, 10) # final output matches num_classes

  def forward(self, x):
    # Conv + ReLU + pool
    out = self.pool(F.relu(self.conv1(x)))
    out = self.pool(F.relu(self.conv2(out)))
    # Flatten it before fc1
    out = out.reshape(-1, 400) # TODO
    out = F.relu(self.fc1(out))
    out = F.relu(self.fc2(out))
    out = F.relu(self.fc3(out))
    out = self.fc4(out) # NO softmax as it will be included in CrossEntropyLoss
    return out


model = CNN().to(device)

# Q13. Use the Cross Entropy Loss for this task (UNCOMMENT & COMPLETE CODE BELOW)
criterion = nn.CrossEntropyLoss()

# Q14. Use the Stochastic Gradient Descent (SGD) optimiser, this time ADD momentum=0.9 (UNCOMMENT & COMPLETE CODE BELOW)
opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
```

## Training loop



```python
n_total_steps = len(train_set)
n_iterations = -(-n_total_steps // batch_size) # ceiling division

for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    #print(images.shape) # [4,3,32,32] batch size, channels, img dim
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass and Optimise
    opt.zero_grad()
    loss.backward()
    opt.step()

    # Print
    if (i+1) % 1000 == 0:
      print(f'Epoch {epoch+1}/{num_epochs}, Iteration {i+1}/{n_iterations}, Loss={loss.item():.4f} ')

```

<pre>
Epoch 1/5, Iteration 1000/12500, Loss=2.3326 
Epoch 1/5, Iteration 2000/12500, Loss=2.2623 
Epoch 1/5, Iteration 3000/12500, Loss=2.0769 
Epoch 1/5, Iteration 4000/12500, Loss=2.0251 
Epoch 1/5, Iteration 5000/12500, Loss=2.3518 
Epoch 1/5, Iteration 6000/12500, Loss=2.0192 
Epoch 1/5, Iteration 7000/12500, Loss=1.6025 
Epoch 1/5, Iteration 8000/12500, Loss=1.2004 
Epoch 1/5, Iteration 9000/12500, Loss=1.7578 
Epoch 1/5, Iteration 10000/12500, Loss=1.7630 
Epoch 1/5, Iteration 11000/12500, Loss=1.6056 
Epoch 1/5, Iteration 12000/12500, Loss=2.0629 
Epoch 2/5, Iteration 1000/12500, Loss=1.0972 
Epoch 2/5, Iteration 2000/12500, Loss=1.9703 
Epoch 2/5, Iteration 3000/12500, Loss=1.1913 
Epoch 2/5, Iteration 4000/12500, Loss=1.8386 
Epoch 2/5, Iteration 5000/12500, Loss=0.4736 
Epoch 2/5, Iteration 6000/12500, Loss=1.0031 
Epoch 2/5, Iteration 7000/12500, Loss=0.5412 
Epoch 2/5, Iteration 8000/12500, Loss=1.5522 
Epoch 2/5, Iteration 9000/12500, Loss=0.8261 
Epoch 2/5, Iteration 10000/12500, Loss=0.9519 
Epoch 2/5, Iteration 11000/12500, Loss=2.1562 
Epoch 2/5, Iteration 12000/12500, Loss=0.5341 
Epoch 3/5, Iteration 1000/12500, Loss=2.4827 
Epoch 3/5, Iteration 2000/12500, Loss=1.5980 
Epoch 3/5, Iteration 3000/12500, Loss=0.9035 
Epoch 3/5, Iteration 4000/12500, Loss=0.1972 
Epoch 3/5, Iteration 5000/12500, Loss=0.5041 
Epoch 3/5, Iteration 6000/12500, Loss=1.1076 
Epoch 3/5, Iteration 7000/12500, Loss=0.9044 
Epoch 3/5, Iteration 8000/12500, Loss=2.0723 
Epoch 3/5, Iteration 9000/12500, Loss=0.2980 
Epoch 3/5, Iteration 10000/12500, Loss=0.6682 
Epoch 3/5, Iteration 11000/12500, Loss=1.9096 
Epoch 3/5, Iteration 12000/12500, Loss=0.3109 
Epoch 4/5, Iteration 1000/12500, Loss=0.5533 
Epoch 4/5, Iteration 2000/12500, Loss=0.6846 
Epoch 4/5, Iteration 3000/12500, Loss=1.0342 
Epoch 4/5, Iteration 4000/12500, Loss=0.9656 
Epoch 4/5, Iteration 5000/12500, Loss=1.3948 
Epoch 4/5, Iteration 6000/12500, Loss=2.8115 
Epoch 4/5, Iteration 7000/12500, Loss=1.4409 
Epoch 4/5, Iteration 8000/12500, Loss=1.9957 
Epoch 4/5, Iteration 9000/12500, Loss=0.9368 
Epoch 4/5, Iteration 10000/12500, Loss=0.7812 
Epoch 4/5, Iteration 11000/12500, Loss=1.2438 
Epoch 4/5, Iteration 12000/12500, Loss=1.0495 
Epoch 5/5, Iteration 1000/12500, Loss=0.0653 
Epoch 5/5, Iteration 2000/12500, Loss=0.6139 
Epoch 5/5, Iteration 3000/12500, Loss=1.2385 
Epoch 5/5, Iteration 4000/12500, Loss=1.0812 
Epoch 5/5, Iteration 5000/12500, Loss=0.8447 
Epoch 5/5, Iteration 6000/12500, Loss=1.5258 
Epoch 5/5, Iteration 7000/12500, Loss=0.9273 
Epoch 5/5, Iteration 8000/12500, Loss=1.8485 
Epoch 5/5, Iteration 9000/12500, Loss=1.1506 
Epoch 5/5, Iteration 10000/12500, Loss=1.1679 
Epoch 5/5, Iteration 11000/12500, Loss=1.8778 
Epoch 5/5, Iteration 12000/12500, Loss=0.9093 
</pre>
## Evaluation



```python
# Deactivate the autograd engine to reduce memory usage and speed up computations (backprop disabled).
with torch.no_grad():
  n_correct = 0
  n_samples = 0
  n_class_correct = [0 for i in range(10)]
  n_class_samples = [0 for i in range(10)]


  # Loop through test set
  for images, labels in test_loader:
    # Put images on GPU
    images = images.to(device)
    labels = labels.to(device)
    # Run on trained model
    outputs = model(images) 

    # Get predictions
    # torch.max() returns actual probability value (ignored) and index or class label (selected)
    _, y_preds = torch.max(outputs, 1)
    n_samples += labels.size(0) # different to FFNN
    n_correct += (y_preds == labels).sum().item()

    # Keep track of each class
    for i in range(batch_size):
      label = labels[i]
      pred = y_preds[i]
      if (label == pred):
        n_class_correct[label] += 1
      n_class_samples[label] += 1

  # Print accuracy
  acc = 100.0 * n_correct / n_samples
  print(f'Test Accuracy of the WHOLE CNN = {acc} %')

  for i in range(len(classes)):
    acc = 100.0 * n_class_correct[i] / n_class_samples[i]
    print(f'Accuracy of {classes[i]}: {acc} %')
```

<pre>
Test Accuracy of the WHOLE CNN = 62.15 %
Accuracy of plane: 67.6 %
Accuracy of car: 80.2 %
Accuracy of bird: 33.0 %
Accuracy of cat: 49.1 %
Accuracy of deer: 71.2 %
Accuracy of dog: 47.2 %
Accuracy of frog: 70.9 %
Accuracy of horse: 61.1 %
Accuracy of ship: 75.1 %
Accuracy of truck: 66.1 %
</pre>

```python
# Q15. Why don't we need to reshape the input images when training and testing?
  ## That's because the shape of train/test images are just fit to the input shape of first convolution layer... I guess.

# Q16. Try to improve the model performance, e.g. by increasing the epochs, changing batch size, adding convolutions, etc.
  ## Changes) num_epochs = 5 / added 1 fc layer
  ## The image below is accuracy of the previous model.
```

![image.png](data:image/png;base64,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)



```python

```
