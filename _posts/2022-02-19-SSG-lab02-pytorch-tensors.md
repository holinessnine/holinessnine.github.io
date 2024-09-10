---
layout: single
title:  "[딥러닝기초] 03. 말로만 듣던 pytorch와 tensor"
categories: DL
tag: [python, deep learning, colab, pytorch]
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


# Lab 2 – Tensors



```python
import torch
```

## Create empty tensors

- Uninitialised tensors



```python
# 1-D
x = torch.empty(1)
print(x)

# 1-D with two elements
x = torch.empty(2)
print(f"\n1-D tensor\n{x}\n")

# 2-D
x = torch.empty(2,3)
print(f"2-D tensor\n{x}\n")

# 3-D
x = torch.empty(2,3,1)
print("3-D tensor\n",x)
```

<pre>
tensor([8.9121e+14])

1-D tensor
tensor([8.9121e+14, 3.0744e-41])

2-D tensor
tensor([[8.9120e+14, 3.0744e-41, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00]])

3-D tensor
 tensor([[[8.9123e+14],
         [3.0744e-41],
         [5.0447e-44]],

        [[0.0000e+00],
         [       nan],
         [0.0000e+00]]])
</pre>
## Initialise tensors to scalar, random values, zeros or ones



```python
# 0-D tensor (just containing scalar value 3)
x = torch.tensor(3)
print(f'Scalar: {x} has shape {x.shape} and {x.ndim} dimensions\n')

# Random values in the interval [0,1]
x = torch.rand(2,3)
print(f"Random values:\n{x}")
print(x.dtype)

# Zeros
x = torch.zeros(2,3)
print(f"\nZeros:\n{x}")
print(x.dtype)

x = torch.zeros(2,3, dtype=torch.int)
print(f"\nZeros:\n{x}\n")

# Ones
x = torch.ones(2,3)
print(f"Ones:\n{x}")
print(f"Type: {x.dtype}\n")

x = torch.ones(2,3, dtype=torch.double)
print(f"Ones:\n{x}")
```

<pre>
Scalar: 3 has shape torch.Size([]) and 0 dimensions

Random values:
tensor([[0.5478, 0.5219, 0.1164],
        [0.0014, 0.3487, 0.3986]])
torch.float32

Zeros:
tensor([[0., 0., 0.],
        [0., 0., 0.]])
torch.float32

Zeros:
tensor([[0, 0, 0],
        [0, 0, 0]], dtype=torch.int32)

Ones:
tensor([[1., 1., 1.],
        [1., 1., 1.]])
Type: torch.float32

Ones:
tensor([[1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
</pre>
## Construct tensor from data



```python
# Change a list to a tensor
x = torch.tensor([0.5, 2.7])
print(x)
```

<pre>
tensor([0.5000, 2.7000])
</pre>
### Changing tensor data type




```python
x = torch.rand(2,3) * 20
print(x)

# Change to integer
y = x.to(torch.int32)
print(y)
```

<pre>
tensor([[ 0.5160, 19.0644,  8.0712],
        [18.3125,  2.8137, 18.2686]])
tensor([[ 0, 19,  8],
        [18,  2, 18]], dtype=torch.int32)
</pre>
### Create a separate copy






```python
a = torch.ones(2, 2)
b = a.clone()

a[0][1] = 561          # a changes...
print(b)               # ...but b is still all ones
```

<pre>
tensor([[1., 1.],
        [1., 1.]])
</pre>
## Basic tensor operations



- addition, subtraction, multiplication, division



```python
x = torch.rand(2,2)
print(f"x = \n{x}\n")

y = torch.rand(2,2)
print(f"y = \n{y}\n")

z = torch.add(x,y) # same as z = x + y
print("x + y = ")
print(z)

z = torch.sub(x,y) # same as z = x - y
print("x - y = ")
print(z)

z = torch.mul(x,y) # same as z = x * y
print("x * y = ")
print(z)

z = torch.div(x,y) # same as z = x / y
print("x / y = ")
print(z)
```

<pre>
x = 
tensor([[0.9046, 0.3864],
        [0.9358, 0.2655]])

y = 
tensor([[0.0076, 0.2886],
        [0.3735, 0.7568]])

x + y = 
tensor([[0.9122, 0.6751],
        [1.3093, 1.0223]])
x - y = 
tensor([[ 0.8970,  0.0978],
        [ 0.5623, -0.4914]])
x * y = 
tensor([[0.0069, 0.1115],
        [0.3495, 0.2009]])
x / y = 
tensor([[119.2523,   1.3387],
        [  2.5055,   0.3508]])
</pre>
### Inplace operations



- any function with a trailing underscore (e.g. ``add_``) will modify the value of the variable in question, in place



```python
x = torch.rand(2,2)
print(f"x = \n{x}\n")

y = torch.rand(2,2)
print(f"y = \n{y}\n")

# Inplace operations
y.add_(x) # modify y by adding x to it
print(f"y + x = {y}")

y.sub_(x) # modify y by subtracting x from it
print(f"y - x = {y}")

y.mul_(x) # modify y by multiplying x to it
print(f"y * x = {y}")

y.div_(x) # modify y by dividing it by x
print(f"y / x = {y}")
```

<pre>
x = 
tensor([[0.7783, 0.6079],
        [0.2100, 0.3334]])

y = 
tensor([[0.3165, 0.1725],
        [0.3060, 0.0965]])

y + x = tensor([[1.0948, 0.7804],
        [0.5160, 0.4299]])
y - x = tensor([[0.3165, 0.1725],
        [0.3060, 0.0965]])
y * x = tensor([[0.2463, 0.1048],
        [0.0643, 0.0322]])
y / x = tensor([[0.3165, 0.1725],
        [0.3060, 0.0965]])
</pre>
## Accessing tensors



```python
# Slicing
x = torch.rand(5,3)
print(x)

# Get all rows but only first column
print(x[:, 0])

# Get all columns but only the second row
print(x[1, :])

# Get a specific element
print(x[2,2])

# When the tensor returns only ONE element, use item() to get the actual value of that element
print(x[1,1].item())

y = torch.tensor([2.0])
print(y.item())
```

<pre>
tensor([[0.4380, 0.9077, 0.2557],
        [0.5748, 0.4927, 0.9742],
        [0.7044, 0.9024, 0.7464],
        [0.3203, 0.9629, 0.2594],
        [0.5469, 0.3166, 0.9898]])
tensor([0.4380, 0.5748, 0.7044, 0.3203, 0.5469])
tensor([0.5748, 0.4927, 0.9742])
tensor(0.7464)
0.4926605224609375
2.0
</pre>
## Tensor Shape & Dimensions

- The number of dimensions a tensor has is called its rank and the length in each dimension describes its ``shape``.

- To determine the length of each dimension, call ``.shape``

- To determine the number of dimensions it has, call ``.ndim``




```python
x = torch.rand(5,3)
print(f'{x} \nhas shape {x.shape} and {x.ndim} dimensions\n')
```

<pre>
tensor([[0.0566, 0.3403, 0.5494],
        [0.1297, 0.3682, 0.1450],
        [0.3965, 0.4230, 0.9653],
        [0.5657, 0.9580, 0.2638],
        [0.3587, 0.1738, 0.1973]]) 
has shape torch.Size([5, 3]) and 2 dimensions

</pre>
## More on Shapes



- `[1,5,2,6]` has shape (4,) to indicate it has 4 elements and the missing element after the comma means it is a 1-D tensor or array (vector)

- `[[1,5,2,6], [1,2,3,1]]` has shape (2,4) to indicate it has 2 elements and each of these have 4 elements. This is a 2-D tensor or array (matrix or a list of vectors)

- `[[[1,5,2,6], [1,2,3,1]], [[5,2,1,2], [6,4,3,2]], [[7,8,5,3], [2,2,9,6]]]` has shape (3, 2, 4) to indicate it has 3 elements in the first dimension, and each of these contain 2 elements and each of these contain 4 elements. This is a 3-D tensor or array



## Operations on tensor dimensions

- A tensor dimension is akin to an array's axis. The number of dimensions is called rank.

- A scalar has rank 0, a vector has rank 1, a matrix has rank 2, a cuboid has rank 3, etc.

- Sometimes one wants to do an operation only on a particular dimension, e.g. on the rows only

- Across ``dim=X`` means we do the operation w.r.t to the dimension given and the rest of the dimensions of the tensor stays as they are

- in 2-D tensors, ``dim=0`` refers to the columns while ``dim=1`` refers to the rows




```python
x = torch.tensor([[1,2,3],
                 [4,5,6]])

print(x.shape)
print(f'Summing across dim=0 (columns) gives: {torch.sum(x,dim=0)}')

print(f'Summing across dim=1 (rows) gives: {torch.sum(x,dim=1)}')
```

<pre>
torch.Size([2, 3])
Summing across dim=0 (columns) gives: tensor([5, 7, 9])
Summing across dim=1 (rows) gives: tensor([ 6, 15])
</pre>
## Reshaping tensors

- There are several ways to do this but using ``torch.reshape()`` is the most common

- Also look up ``torch.squeeze(), torch.unsqueeze()`` and ``torch.view()``




```python
x = torch.rand(4,4)
print("Original:")
print(x)
print(x.shape)

# Reshape (flatten) to 1-D
y = x.reshape(16) # number of elements must be the same as original, error otherwise
print("Reshaped to 1-D:")
print(y)

# Reshape to 2-D
y = x.reshape(8,2)
print("Reshaped to 2-D:")
print(y)

# Could leave out one of the dimensions by specifying -1
y = x.reshape(2, -1)
print("Reshaped to 2 x Unspecified 2-D:")
print(y)
print(y.shape)

# Could use unsqueeze(0) to add a dimension at position 0
y = x.unsqueeze(0)
print(f'Using unsqueeze(0) to add dimension from original shape {x.shape} to {y.shape}')
```

<pre>
Original:
tensor([[0.4850, 0.5294, 0.2495, 0.4379],
        [0.2554, 0.6912, 0.0444, 0.6933],
        [0.2610, 0.9324, 0.3359, 0.5522],
        [0.7893, 0.7543, 0.0788, 0.2166]])
torch.Size([4, 4])
Reshaped to 1-D:
tensor([0.4850, 0.5294, 0.2495, 0.4379, 0.2554, 0.6912, 0.0444, 0.6933, 0.2610,
        0.9324, 0.3359, 0.5522, 0.7893, 0.7543, 0.0788, 0.2166])
Reshaped to 2-D:
tensor([[0.4850, 0.5294],
        [0.2495, 0.4379],
        [0.2554, 0.6912],
        [0.0444, 0.6933],
        [0.2610, 0.9324],
        [0.3359, 0.5522],
        [0.7893, 0.7543],
        [0.0788, 0.2166]])
Reshaped to 2 x Unspecified 2-D:
tensor([[0.4850, 0.5294, 0.2495, 0.4379, 0.2554, 0.6912, 0.0444, 0.6933],
        [0.2610, 0.9324, 0.3359, 0.5522, 0.7893, 0.7543, 0.0788, 0.2166]])
torch.Size([2, 8])
Using unsqueeze(0) to add dimension from original shape torch.Size([4, 4]) to torch.Size([1, 4, 4])
</pre>
## Convert between NumPy and PyTorch tensors



- Tensors can work on CPUs and GPUs

- NumPy arrays can only work on CPUs



```python
import torch
import numpy as np

# Tensor to NumPy
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)
print(type(b))

# b changes when a is modified because they share the same memory space!
a.add_(1)
print(a)
print(b)
```

<pre>
tensor([1., 1., 1., 1., 1.])
[1. 1. 1. 1. 1.]
<class 'numpy.ndarray'>
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]
</pre>

```python
import torch
import numpy as np

a = np.ones(5)
print(a)

b = torch.from_numpy(a)
print(b)

# Modifying array will modify the tensor as well
a += 2
print(a)
print(b)
```

<pre>
[1. 1. 1. 1. 1.]
tensor([1., 1., 1., 1., 1.], dtype=torch.float64)
[3. 3. 3. 3. 3.]
tensor([3., 3., 3., 3., 3.], dtype=torch.float64)
</pre>
# Exercise



```python
# 1. Create a tensor of 100 equally spaced numbers from 0 to 2.
# Assign the tensor to x (Hint: use torch.linspace() )
x = torch.linspace(start = 0, end = 2, steps = 100)
# Print x
print(x)
```

<pre>
tensor([0.0000, 0.0202, 0.0404, 0.0606, 0.0808, 0.1010, 0.1212, 0.1414, 0.1616,
        0.1818, 0.2020, 0.2222, 0.2424, 0.2626, 0.2828, 0.3030, 0.3232, 0.3434,
        0.3636, 0.3838, 0.4040, 0.4242, 0.4444, 0.4646, 0.4848, 0.5051, 0.5253,
        0.5455, 0.5657, 0.5859, 0.6061, 0.6263, 0.6465, 0.6667, 0.6869, 0.7071,
        0.7273, 0.7475, 0.7677, 0.7879, 0.8081, 0.8283, 0.8485, 0.8687, 0.8889,
        0.9091, 0.9293, 0.9495, 0.9697, 0.9899, 1.0101, 1.0303, 1.0505, 1.0707,
        1.0909, 1.1111, 1.1313, 1.1515, 1.1717, 1.1919, 1.2121, 1.2323, 1.2525,
        1.2727, 1.2929, 1.3131, 1.3333, 1.3535, 1.3737, 1.3939, 1.4141, 1.4343,
        1.4545, 1.4747, 1.4949, 1.5152, 1.5354, 1.5556, 1.5758, 1.5960, 1.6162,
        1.6364, 1.6566, 1.6768, 1.6970, 1.7172, 1.7374, 1.7576, 1.7778, 1.7980,
        1.8182, 1.8384, 1.8586, 1.8788, 1.8990, 1.9192, 1.9394, 1.9596, 1.9798,
        2.0000])
</pre>

```python
# 3. Print the first 5 numbers in x.
print(x[:5])
# 4. Print the last 5 numbers in x
print(x[-5:])
```


```python
# 5. Create another tensor of 100 random values between 0 and 1.
# Assign the tensor to y (Hint: use torch.rand() )
y = torch.rand(100)
# Print y
print(y)
```

<pre>
tensor([0.8459, 0.9065, 0.6888, 0.1062, 0.0362, 0.3217, 0.8137, 0.3431, 0.5249,
        0.5184, 0.0224, 0.1610, 0.6187, 0.9712, 0.1306, 0.6729, 0.1973, 0.6566,
        0.2996, 0.9720, 0.0037, 0.1647, 0.4556, 0.7612, 0.7076, 0.9992, 0.6992,
        0.4529, 0.6154, 0.4097, 0.2055, 0.1894, 0.9545, 0.5765, 0.8383, 0.5207,
        0.3411, 0.7809, 0.4343, 0.0607, 0.2502, 0.5164, 0.4957, 0.1078, 0.6526,
        0.2413, 0.2064, 0.0869, 0.3961, 0.0253, 0.8101, 0.0625, 0.1069, 0.8273,
        0.9331, 0.2262, 0.6626, 0.4722, 0.6993, 0.2771, 0.6754, 0.7479, 0.8137,
        0.0384, 0.3906, 0.9565, 0.8528, 0.9466, 0.9299, 0.6296, 0.4327, 0.8660,
        0.4907, 0.4048, 0.7270, 0.3851, 0.4518, 0.6275, 0.3287, 0.7962, 0.7089,
        0.8551, 0.4690, 0.6506, 0.4791, 0.4607, 0.6427, 0.4246, 0.6596, 0.2928,
        0.2214, 0.7380, 0.0427, 0.3041, 0.2161, 0.7893, 0.4225, 0.7324, 0.4951,
        0.5642])
</pre>

```python
# 6. Multiply x and y, store the result in z
z = torch.mul(x,y)

# Print z
print(z)
```

<pre>
tensor([0.0000e+00, 1.8313e-02, 2.7831e-02, 6.4351e-03, 2.9256e-03, 3.2495e-02,
        9.8631e-02, 4.8517e-02, 8.4840e-02, 9.4254e-02, 4.5198e-03, 3.5769e-02,
        1.4999e-01, 2.5506e-01, 3.6946e-02, 2.0391e-01, 6.3764e-02, 2.2550e-01,
        1.0893e-01, 3.7309e-01, 1.4872e-03, 6.9856e-02, 2.0249e-01, 3.5368e-01,
        3.4310e-01, 5.0464e-01, 3.6727e-01, 2.4703e-01, 3.4812e-01, 2.4004e-01,
        1.2454e-01, 1.1864e-01, 6.1707e-01, 3.8433e-01, 5.7581e-01, 3.6817e-01,
        2.4807e-01, 5.8371e-01, 3.3342e-01, 4.7812e-02, 2.0216e-01, 4.2774e-01,
        4.2062e-01, 9.3675e-02, 5.8010e-01, 2.1934e-01, 1.9180e-01, 8.2532e-02,
        3.8410e-01, 2.5086e-02, 8.1829e-01, 6.4355e-02, 1.1233e-01, 8.8575e-01,
        1.0179e+00, 2.5129e-01, 7.4963e-01, 5.4372e-01, 8.1934e-01, 3.3030e-01,
        8.1870e-01, 9.2171e-01, 1.0192e+00, 4.8827e-02, 5.0507e-01, 1.2561e+00,
        1.1371e+00, 1.2813e+00, 1.2774e+00, 8.7757e-01, 6.1187e-01, 1.2422e+00,
        7.1368e-01, 5.9699e-01, 1.0868e+00, 5.8344e-01, 6.9371e-01, 9.7615e-01,
        5.1795e-01, 1.2707e+00, 1.1457e+00, 1.3992e+00, 7.7700e-01, 1.0909e+00,
        8.1305e-01, 7.9116e-01, 1.1166e+00, 7.4628e-01, 1.1727e+00, 5.2643e-01,
        4.0257e-01, 1.3568e+00, 7.9440e-02, 5.7137e-01, 4.1044e-01, 1.5148e+00,
        8.1945e-01, 1.4353e+00, 9.8012e-01, 1.1284e+00])
</pre>

```python
# 7. Reshape z to a tensor with 5 rows and 20 columns
# Store reshaped tensor to z2
z2 = z.reshape(5, 20)

# Print z2
print(z2)
z2.shape
```

<pre>
tensor([[0.0000e+00, 1.8313e-02, 2.7831e-02, 6.4351e-03, 2.9256e-03, 3.2495e-02,
         9.8631e-02, 4.8517e-02, 8.4840e-02, 9.4254e-02, 4.5198e-03, 3.5769e-02,
         1.4999e-01, 2.5506e-01, 3.6946e-02, 2.0391e-01, 6.3764e-02, 2.2550e-01,
         1.0893e-01, 3.7309e-01],
        [1.4872e-03, 6.9856e-02, 2.0249e-01, 3.5368e-01, 3.4310e-01, 5.0464e-01,
         3.6727e-01, 2.4703e-01, 3.4812e-01, 2.4004e-01, 1.2454e-01, 1.1864e-01,
         6.1707e-01, 3.8433e-01, 5.7581e-01, 3.6817e-01, 2.4807e-01, 5.8371e-01,
         3.3342e-01, 4.7812e-02],
        [2.0216e-01, 4.2774e-01, 4.2062e-01, 9.3675e-02, 5.8010e-01, 2.1934e-01,
         1.9180e-01, 8.2532e-02, 3.8410e-01, 2.5086e-02, 8.1829e-01, 6.4355e-02,
         1.1233e-01, 8.8575e-01, 1.0179e+00, 2.5129e-01, 7.4963e-01, 5.4372e-01,
         8.1934e-01, 3.3030e-01],
        [8.1870e-01, 9.2171e-01, 1.0192e+00, 4.8827e-02, 5.0507e-01, 1.2561e+00,
         1.1371e+00, 1.2813e+00, 1.2774e+00, 8.7757e-01, 6.1187e-01, 1.2422e+00,
         7.1368e-01, 5.9699e-01, 1.0868e+00, 5.8344e-01, 6.9371e-01, 9.7615e-01,
         5.1795e-01, 1.2707e+00],
        [1.1457e+00, 1.3992e+00, 7.7700e-01, 1.0909e+00, 8.1305e-01, 7.9116e-01,
         1.1166e+00, 7.4628e-01, 1.1727e+00, 5.2643e-01, 4.0257e-01, 1.3568e+00,
         7.9440e-02, 5.7137e-01, 4.1044e-01, 1.5148e+00, 8.1945e-01, 1.4353e+00,
         9.8012e-01, 1.1284e+00]])
</pre>
<pre>
torch.Size([5, 20])
</pre>

```python
# 8. Get the sum of each row in z2
print(f'Sum of each row in z2: {torch.sum(z2, dim=1)}\n')

# 9. Get the mean of each column in z3
print(f'Mean of each column in z2: {torch.mean(z2, dim=0)}')
```

<pre>
Sum of each row in z2: tensor([ 1.8717,  6.0793,  8.2201, 17.4364, 18.2777])

Mean of each column in z2: tensor([0.4336, 0.5674, 0.4894, 0.3187, 0.4489, 0.5607, 0.5823, 0.4811, 0.6534,
        0.3527, 0.3924, 0.5635, 0.3345, 0.5387, 0.6256, 0.5843, 0.5149, 0.7529,
        0.5520, 0.6301])
</pre>

```python
# 10. Reshape z to a 3D tensor (keep all the elements)
# Store reshaped tensor to z3
z3 = z.reshape(5, 4, -1)
# Print z3
print(z3)
```

<pre>
tensor([[[0.0000e+00, 1.8313e-02, 2.7831e-02, 6.4351e-03, 2.9256e-03],
         [3.2495e-02, 9.8631e-02, 4.8517e-02, 8.4840e-02, 9.4254e-02],
         [4.5198e-03, 3.5769e-02, 1.4999e-01, 2.5506e-01, 3.6946e-02],
         [2.0391e-01, 6.3764e-02, 2.2550e-01, 1.0893e-01, 3.7309e-01]],

        [[1.4872e-03, 6.9856e-02, 2.0249e-01, 3.5368e-01, 3.4310e-01],
         [5.0464e-01, 3.6727e-01, 2.4703e-01, 3.4812e-01, 2.4004e-01],
         [1.2454e-01, 1.1864e-01, 6.1707e-01, 3.8433e-01, 5.7581e-01],
         [3.6817e-01, 2.4807e-01, 5.8371e-01, 3.3342e-01, 4.7812e-02]],
    
        [[2.0216e-01, 4.2774e-01, 4.2062e-01, 9.3675e-02, 5.8010e-01],
         [2.1934e-01, 1.9180e-01, 8.2532e-02, 3.8410e-01, 2.5086e-02],
         [8.1829e-01, 6.4355e-02, 1.1233e-01, 8.8575e-01, 1.0179e+00],
         [2.5129e-01, 7.4963e-01, 5.4372e-01, 8.1934e-01, 3.3030e-01]],
    
        [[8.1870e-01, 9.2171e-01, 1.0192e+00, 4.8827e-02, 5.0507e-01],
         [1.2561e+00, 1.1371e+00, 1.2813e+00, 1.2774e+00, 8.7757e-01],
         [6.1187e-01, 1.2422e+00, 7.1368e-01, 5.9699e-01, 1.0868e+00],
         [5.8344e-01, 6.9371e-01, 9.7615e-01, 5.1795e-01, 1.2707e+00]],
    
        [[1.1457e+00, 1.3992e+00, 7.7700e-01, 1.0909e+00, 8.1305e-01],
         [7.9116e-01, 1.1166e+00, 7.4628e-01, 1.1727e+00, 5.2643e-01],
         [4.0257e-01, 1.3568e+00, 7.9440e-02, 5.7137e-01, 4.1044e-01],
         [1.5148e+00, 8.1945e-01, 1.4353e+00, 9.8012e-01, 1.1284e+00]]])
</pre>

```python

```
