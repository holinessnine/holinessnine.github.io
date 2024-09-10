---
layout: single
title:  "[딥러닝기초] 01. colab을 시작해보자!"
categories: DL
tag: [python, deep learning, colab]
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


# Lab 1 – Colab basics 

### NOTE: Before starting this tutorial

- This file is not editable. **Save a copy to your Drive** first before working on it:

  - ``File-->Save a copy in Drive``

- The copy of the notebook should open and you can work on it. 

- Click ``Connect-->Connect to a hosted runtime``

- **runtime** refers to the hosted virtual machine on Google Cloud

- **local machine** refers to your own machine, e.g. PC or laptop



## 1. Command lines in Colab



- Command lines are preceeded by a ``!``

- For more command lines that can help you navigate within your working space, see [here](http://www.mathcs.emory.edu/~valerie/courses/fall10/155/resources/unix_cheatsheet.html)

- Run each of these code cells and understand what it is doing.



```python
# Show current directory
!pwd

# See the contents of the current directory
!ls -al

# Verify this by clicking on the "Folder" image on the left to see the current directory pictorially
# Can you see what is inside the folder "sample_data" by clicking on it?
# Command line alternative: how would you modify the ls command to see the 
# list of files inside "sample_data"?
!ls -al sample_data
```

<pre>
/content
total 16
drwxr-xr-x 1 root root 4096 Aug 31 13:18 .
drwxr-xr-x 1 root root 4096 Sep  2 03:57 ..
drwxr-xr-x 4 root root 4096 Aug 25 13:35 .config
drwxr-xr-x 1 root root 4096 Aug 31 13:18 sample_data
total 55516
drwxr-xr-x 1 root root     4096 Aug 31 13:18 .
drwxr-xr-x 1 root root     4096 Aug 31 13:18 ..
-rwxr-xr-x 1 root root     1697 Jan  1  2000 anscombe.json
-rw-r--r-- 1 root root   301141 Aug 31 13:18 california_housing_test.csv
-rw-r--r-- 1 root root  1706430 Aug 31 13:18 california_housing_train.csv
-rw-r--r-- 1 root root 18289443 Aug 31 13:18 mnist_test.csv
-rw-r--r-- 1 root root 36523880 Aug 31 13:18 mnist_train_small.csv
-rwxr-xr-x 1 root root      930 Jan  1  2000 README.md
</pre>
## Changing directory

- You cannot use ``!cd`` to navigate the filesystem because shell commands in the notebook are executed in a temporary subshell.

- In general, use ``!`` if the command is one that's okay to run in a separate shell. Use ``%`` if the command needs to be run on the specific notebook.



```python
# Try to go one directory up
!cd ..

!pwd # Does it go to the root directory or remains the same as before?
```

<pre>
/content
</pre>
Now try use the ``%cd`` magic command:




```python
# Go one directory up
%cd ..

!pwd
```

<pre>
/
/
</pre>
Go back to the previous directory you were in



```python
%cd -
```

<pre>
/content
</pre>
## 2. Processor related commands

- Check if GPU is available by running the cell below




```python
import torch 

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("GPU")
else:
  device = torch.device("cpu")
  print("CPU")
```

<pre>
GPU
</pre>
### Get further info on processor



```python
# Gets CPU info on the virtual machine
!cat /proc/cpuinfo | grep 'processor\|model name\|cpu cores'
```

<pre>
processor	: 0
model name	: Intel(R) Xeon(R) CPU @ 2.30GHz
cpu cores	: 1
processor	: 1
model name	: Intel(R) Xeon(R) CPU @ 2.30GHz
cpu cores	: 1
</pre>

```python
# Get GPU inf on the virtual machine
!nvidia-smi
```

<pre>
Thu Sep  2 03:58:25 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.57.02    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla K80           Off  | 00000000:00:04.0 Off |                    0 |
| N/A   48C    P8    31W / 149W |      3MiB / 11441MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
</pre>
## REPEAT the last three cells

- After running the 3 code cells, click ``Runtime-->Change runtime type-->GPU``

and rerun them now. Do you see different outcomes now?



## 3. Python related command lines



```python
!python --version
```

<pre>
Python 3.7.11
</pre>
To import a library that's not in Colaboratory by default, you can use ``!pip install`` or ``!apt-get install``



```python
!pip install matplotlib-venn
```

<pre>
Requirement already satisfied: matplotlib-venn in /usr/local/lib/python3.7/dist-packages (0.11.6)
Requirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from matplotlib-venn) (1.4.1)
Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from matplotlib-venn) (1.19.5)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from matplotlib-venn) (3.2.2)
Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->matplotlib-venn) (2.4.7)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->matplotlib-venn) (1.3.1)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->matplotlib-venn) (0.10.0)
Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->matplotlib-venn) (2.8.2)
Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from cycler>=0.10->matplotlib->matplotlib-venn) (1.15.0)
</pre>
Show information about one or more installed packages.



```python
!pip show torch
```

<pre>
Name: torch
Version: 1.9.0+cu102
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page: https://pytorch.org/
Author: PyTorch Team
Author-email: packages@pytorch.org
License: BSD-3
Location: /usr/local/lib/python3.7/dist-packages
Requires: typing-extensions
Required-by: torchvision, torchtext, fastai
</pre>
Python printing



```python
print('This is a sample sentence')
```

<pre>
This is a sample sentence
</pre>

```python
x = 5

print(f"The value of x is {x}")
```

<pre>
The value of x is 5
</pre>
## 4. Working with Files

- Mount Drive onto the runtime machine so that you can load files from Drive to your runtime

- NOTE: You will most likely be asked for an authorisation code, click on the link and copy the code over

- ~~You could also mount Drive by clicking on on the Mount Drive button on the left panel~~




```python
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
```

<pre>
Mounted at /content/gdrive
</pre>
### If you click on the "Folder" image on the left, you should see an additional folder "gdrive"

- List content of mounted drive



```python
!ls '/content/gdrive'
```

<pre>
MyDrive  Shareddrives
</pre>
### What about loading files from your local machine?

- Run the code below, you should be able to choose one or more files to be uploaded to the runtime

- Press 'Choose files' and select the file(s) that you want to upload

- Verify by seeing the runtime's folder content on the left



```python
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
  
```


     <input type="file" id="files-2e1e3cff-e22a-4644-a3c6-57ff80736301" name="files[]" multiple disabled
        style="border:none" />
     <output id="result-2e1e3cff-e22a-4644-a3c6-57ff80736301">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script src="/nbextensions/google.colab/files.js"></script> 


## Downloading files from runtime to the local machine

- Create a new file (or overwrite an existing file) on runtime and download it to local

- <code>files.download</code> will invoke a browser download of the file to your local computer.




```python
from google.colab import files

# Select the file from the runtime to local
with open('example.txt', 'w') as f:
  f.write('This is a simple file with one line of text.')
  f.close()

files.download('example.txt')
```

Download an existing file from runtime to local



```python
# Existing file on runtime to local
open('sample_data/README.md', 'r')

files.download('sample_data/README.md')
```

# 5. Working with images

### Upload file from mounted folder and view it

- Import PIL (Python Imaging Library) for image manipulation

- Import matplotlib for visualisation

- Save an image file (png or jpg in a folder in Drive)

- Make sure Drive is mounted to the runtime after the image is saved

- Open the file using Image.open()

- Display the image



```python
from PIL import Image
import matplotlib.pyplot as plt

# 1. Locate the file in Drive and open it
img = Image.open("gdrive/MyDrive/DL/two_small.png") # TODO: replace this with a path to an image in your own Drive

plt.imshow(img)
```
