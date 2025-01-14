---
layout: single
title:  "[딥러닝기초] 02. colab으로 끄적여보는 numpy"
categories: DL
tag: [python, deep learning, colab, numpy]
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




# Lab 1 – NumPy 

---



NumPy is a library in Python for processing multi-dimensional arrays, along with a large collection of high-level mathematical functions to operate on these arrays.






```python
import numpy as np

print("Numpy:", np.__version__)
```

### Array creation



Create a simple scalar as an array




```python
a = np.array(5)
print(a.ndim)
```

Create a simple 1-D array




```python
a = np.array([0,1,2])
print(a.size)
```

Create a 1-D array with zeros in it



```python
a = np.zeros(3)

print(a)
print(type(a))
print(type(a[0])) # zeros are floats
print(a.ndim)
print(a.dtype)
```

Check array properties



```python
# print array type
print(type(a))
# print type of array's first element
print(type(a[0])) # zeros are floats
# print number of dimensions of array
print(a.ndim)
# print array data type
print(a.dtype)
```

Specifying array data type



```python
a = np.array([1,2,3], dtype='int32')
print(a.dtype)
```

Create an array with 1's in it



```python
z = np.ones(10)
z
```

Specying 2-D array data type




```python
z = np.ones((2,4), dtype=int) # Default dtype is numpy.float64
print(z)
print(z.dtype)
```

Fill array with a number



```python
np.full((3,4), 7)
```

### Identity matrix



```python
np.identity(3)
```


```python
np.eye(2)
```

## Array Shape



The shape of an array is the number of elements in each dimension.



```python
a = np.array([0,1,2])
print(a.shape)
```

Array rank is the number of dimensions



```python
print(a.ndim)
```

### Reshape

By reshaping we can add or remove dimensions or change number of elements in each dimension.



```python
a.shape = (3,1)
print(a)
```

Reshape from 1-D to 2-D



```python
b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
c = b.reshape(4, 3)
print(c)
```

Reshape to 3-D

- Arrays can be reshaped to any dimensions as long as the number of elements match




```python
d = c.reshape(2,3,2)
print(d)
d.shape
```

Unknown dimension



- You are allowed to have ONE "unknown" dimension *only*.

- Specify -1 for this, NumPy will calculate this number automatically.



```python
e = c.reshape(1,2,-1)
print(e)
e.shape
```

Reshape back to 1-D



```python
e.reshape(12)
```

## Evenly spaced numbers over a specified number of elements or interval

- ``linspace()``

- ``arange()``




```python
z = np.linspace(2, 10, 5) # from 2 to 10 with 5 elements
```


```python
z = np.arange(0, 10, step=5) # from 0 to 9 with step size 5
z
```

<pre>
array([0, 5])
</pre>

```python
z = np.arange(10)
z
```



# NumPy Arrays vs Lists



*   Faster to read because it uses less bytes of memory via fixed type (e.g. Int32)

*   No type checking when iterating through objects

*   Utilises contiguous memory

*   Element wise operations possible




```python
# Create array from list
a = np.array([1,3,7])
print(a)
type(a)
```

Convert array to list



```python
alist = np.ones(5).tolist()
print(alist)
type(alist)
```

## 2-D arrays



```python
alist = [[1,2,3,4], [8,7,6,5]]
z = np.array(alist)
print(z)
print(z.shape)
```


```python
blist = [[1,2,3,4], [8,7,6,5]]
z = np.array([blist])
print(z)
print(z.shape)
```

## Random Array



Create a random integer array specifying range and size



```python
# random integers start, end exclusive, shape
np.random.randint(4,9,size=(3,4))
```

Create an array with 10 random numbers between 0 and 1



```python
np.random.random(10)
```

Use seed() for reproducibility



```python
# Seed allows the random values generated will be the same every time the code is run
np.random.seed(0) # change 0 to any other integer value (e.g. 10) and see the effects
z1 = np.random.randint(10, size=6) # generate an array with 6 integers between 0 and 9
z1
```

## Basic operations



```python
z = np.array([1,2,3,8,9])
print(z<3)
z[z>3]
```


```python
a = np.array([1,2,3,4,5])
b = np.array([6,7,8,9,10])
print(a+b)
print(a-b)
print(a*b)
print(a/b)
print(a * 10)
```

## Dot product



```python
a @ b
```

## Transpose



Swap the rows and columns of an array



```python
c = np.array([[1,2,3], [4,5,6]])
c.T
```

## Indexing & Slicing



- Remember indexes start at 0

- Accessing specific elements



```python
c = np.array([[1,2,3,4,5,6,7], [8,9,10,11,12,13,14]])

# Index starts at 0
print(c[0,0])
print(c[1,2]) # row 1 (2nd row) , column 2 (3rd column)
```

## Slicing

- ``a[start:stop]``  # items ``start`` through ``stop-1``

- ``a[start:]``      # items ``start`` through the rest of the array

- ``a[:stop]``       # items from the beginning through ``stop-1``

- ``a[:]``           # a copy of the whole array




```python
# Get specific row/column
print(c[0, : ]) # get row 0
print(c[ : ,2]) # get column 2
print(c.shape)
```

### Indexing from the back

- ``a[-1]``    # last item in the array

- ``a[-2:]``   # last two items in the array

- ``a[:-2]``   # everything except the last two items

- Backward indexing starts from -1



```python
b = np.array([[9.0, 8.0, 7.0, 6.0], [5.0, 4.0, 3.0, 2.0]])
print(b[1,-2]) # starting indexing backwards (second element from the back)
```

### Including step

- ``a[start:stop:step]`` # ``start`` through not past ``stop``, by ``step``



```python
print(c[0,1:6:2])
print(c[1,1:-1:2])
```

### Changing elements



> Indented block






```python
c[1,5] = 20
print(c)

c[:, 2] = [17,19]
print(c)
```

### Indexing 3-D arrays



```python
d = np.array([[[1,2],[3,4]], [[5,6],[7,8]]])
print(d)
print(d.shape)
```


```python
# Get specific element
print(d[0,1,1])

# All x, some y , all z
print(d[:,1,:])
```


```python
# Replace
d[:,1,:] = [[9,9], [8,8]]
d
```

## Repeat array



```python
e = np.array([1,2,3])
np.repeat(e, 2)
```


```python
# repeat on an axis
e = np.array([[1,2,3]])
np.repeat(e, 2, axis=0)
```


```python
f = np.ones((5,5), dtype='int32')
f[1:-1, 1:-1] = 0
f[2,2] = 9
f
```

## Making copies




```python
g = np.array([1,2,3])
h = g # point h to g so they share content
h[1] = 100
print(g)
```


```python
g = np.array([1,2,3])
h = g.copy() # explicitly make a copy
h[1] = 100
print(g)
```

# Linear Algebra



```python
a = np.ones((2,3))
print(a)

b = np.full((3,2),2)
print(b)

np.matmul(a,b)
```

# Statistics



```python
a = np.array([[1,2,3], [4,5,6]])
print(a.sum())   
print(a.mean()) 
print(a.max())   
print(a.min())   
print(a.argmax())
```

Specifying axis

- Column: ``axis=0``

- Row: ``axis=1``



```python
print(np.min(a, axis=1)) # smallest values in rows
print(np.max(a, axis=0)) # largest values in columns
```

Others – Log and exponential



```python
print(np.log(a)) 
print(np.exp(a)) 
```

## Vertical stacking



```python
v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])

np.vstack([v1,v2,v1,v2])
```


```python
np.hstack([v2,v1,v1])
```

<pre>
array([5, 6, 7, 8, 1, 2, 3, 4, 1, 2, 3, 4])
</pre>

```python
h1 = np.zeros((2,4))
h2 = np.ones((2,2))

np.hstack((h1,h2))
```

## Plotting with Matplotlib

- A simple example using post office international parcel charges for two countries



```python
import matplotlib.pyplot as plt
import numpy as np

# Parcel weights start at 0.5kg to 20.5kgs in half kilo intervals
x = np.arange(0.5, 20.5, 0.5)

# Country 1 parcel charges
c1 = np.array([16.0, 19.0, 21.5, 24.5, 25.0, 27.3, 29.6, 31.9, 34.2, 36.5, 38.8, 41.1, 43.4, 45.7, 48.0, 50.3, 52.6, 54.9, 57.2, 59.5, 
     61.8, 64.1, 66.4, 68.7, 71.0, 73.3, 75.6, 77.9, 80.2, 82.5, 84.8, 87.1, 89.4, 91.7, 94.0, 96.3, 98.6, 100.9, 103.2, 105.5])

# Country 2 parcel charges
c2 = np.array([16.0, 19.0, 21.5, 24.5, 25.5, 27.0, 28.5, 30.0, 31.5, 33, 34.5, 36, 37.5, 39, 40.5, 42, 43.5, 45, 46.5, 48, 
     49.5, 51, 52.5, 54, 55.5, 57, 58.5, 60, 61.5, 63, 64.5, 66, 67.5, 69, 70.5, 72, 73.5, 75, 76.5, 78])

plt.plot(x,c1)
plt.plot(x,c2)
plt.xlabel("Weight (kg)")
plt.ylabel("Price (Thousand Won)")
plt.legend(['Country 1', 'Country 2'])
plt.show()
```

<img src="data:image/png;base64,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"/>


```python
# Get estimated value from plot using np.interp() for postal price for a weight that's not in the array
parcel_weight = 11.3 # change this to any other value and run again

est1 = np.interp(parcel_weight, x, c1)

est2 = np.interp(parcel_weight, x, c2)

print(f'Estimated parcel prices for weight {parcel_weight:.2f} is {est1} for Country 1 and {est2:.2f} for Country 2')
```
