#What is Numpy

'''Numpy is a library for multi-dimensional arrays and matrices, along with a 
large collection of high-level mathematical functions to operate on these arrays 
and matrices.'''

#Numpy V/S PyList

'''Numpy consumes less memory than PyList
Numpy is faster than PyList
List[1,2,3,4]  Array[1 2 3 4]'''

#Installation and Import

'''pip install numpy
import numpy as np'''

#List and Numpy Array

import numpy as np
x = np.array([1,2,3,4])     #1D Array
print(x)    # [1 2 3 4]              
print(type(x))  # <class 'numpy.ndarray'>

y = [1,2,3,4,5]     #list
print(y)    # [1, 2, 3, 4, 5]
print(type(y))  # <class 'list'>

#Creating Numpy Arrays

'''To create a Numpy array we need to import the numpy module and then use the 
array() function to create an array from a list, tuple, or other array object.'''

a = np.array([1,2,3])
print(a)    # [1 2 3]

l = []
for i in range(1,11):
    l.append(i)

b = np.array(l)
print(b)    # [ 1  2  3  4  5  6  7  8  9 10]

'''NOTE: Use ndim function to find the dimension of an array'''

#To create n dimension array

arn = np.array([1,2,3], ndmin = 10)
print(arn)  # [1 2 3]
print(arn.ndim)  # 10

#Zeros in array Numpy

'''To create an array of zeros, we can use the zeros() function'''

z = np.zeros(10)
print(z)    # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

z1 = np.zeros((3,4))
print(z1)    # [[0. 0. 0. 0.] [0. 0. 0. 0.] [0. 0. 0. 0.]]

#Ones in array Numpy

'''To create an array of ones, we can use the ones() function'''

o = np.ones(10)
print(o)    # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]

o1 = np.ones((3,4))
print(o1)    # [[1. 1. 1. 1.] [1. 1. 1. 1.] [1. 1. 1. 1.]]

#Empty Numpy Array

'''To create an empty array, we can use the empty() function'''

e = np.empty(10)
print(e)    # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]

e1 = np.empty((3,4))
print(e1)    # [[1. 1. 1. 1.] [1. 1. 1. 1.] [1. 1. 1. 1.]]

'''Empty or previous output will be stored'''

#Array with range of elements

'''To create an array with a range of elements, we can use the arange() function'''

ar = np.arange(10)
print(ar)    # [0 1 2 3 4 5 6 7 8 9]

ar1 = np.arange(1,11)
print(ar1)    # [ 1  2  3  4  5  6  7  8  9 10]

ar2 = np.arange(1,11,2)
print(ar2)    # [1 3 5 7 9]

#Array diagonal element filled with 1's

'''To create a diagonal array, we can use the eye() function'''

d = np.eye(3)
print(d)    # [[1. 0. 0.] [0. 1. 0.] [0. 0. 1.]]

d1 = np.eye(3,4)
print(d1)    # [[1. 0. 0. 0.] [0. 1. 0. 0.] [0. 0. 1. 0.]]

#Linear space array

'''To create a linear space array, we can use the linspace() function'''

l = np.linspace(1,10,5)
print(l)    # [ 1.   3.25  5.5  7.75 10.]

l1 = np.linspace(1,10,5, endpoint = False)
print(l1)    # [1.  3.  5.  7.  9.]

#Create Numpy arrays with Random numbers

'''rand(): Function is used to generate a random value between 0 and 1'''

var = np.random.rand(5)
print(var)  # [0.808302 0.231975 0.087237 0.624403 0.979979]

'''randn(): Function is used to generate a random value close to zero. This may
return positive or negative no as well'''

var1 = np.random.randn(5)
print(var1) # [-0.498498 0.498498 0.498498 0.498498 0.498498]

'''randf(): returns an array of specified shape and fills it with random floats in 
half open interval'''

var2 = np.random.randf(5)
print(var2) # [0.808302 0.231975 0.087237 0.624403 0.979979]

'''randint(): used to generate random number between a given range'''

var3 = np.random.randint(1,10,5)
print(var3) # [4 5 6 7 8]

#Identify data type in Numpy Array

'''To identify the data type of an array, we can use the dtype attribute'''

a = np.array([1,2,3,4])
print(a.dtype)  # int32

b = np.array([1.1,2.2,3.3,4.4])
print(b.dtype)  # float64

c = np.array(['a','b','c','d'])
print(c.dtype)  # <U1

d = np.array([True, False, True])
print(d.dtype)  # bool

d = np.array([True, "S", 1, 2.4])
print(d.dtype)  # object

#Change data type of var

'''To change the data type of an array, we can use the astype() function'''

a = np.array([1,2,3,4])
print(a)    # [1 2 3 4]
print(a.dtype)  # int32

b = a.astype(np.float64)
print(b)    # [1. 2. 3. 4.]
print(b.dtype)  # float64

#Arithmetic Operations

'''To perform arithmetic operations on arrays, we can use the following functions:'''

a = np.array([1,2,3,4])
b = np.array([5,6,7,8])
print(a + b)    # [ 6  8 10 12]
print(a - b)    # [-4 -4 -4 -4]
print(a * b)    # [ 5 12 21 32]
print(a / b)    # [0.2 0.33333333 0.42857143 0.5]

#Arithmetic Functions

'''Numpy provides a wide range of arithmetic functions to perform operations on 
arrays. Some of the most commonly used functions are:'''

var = np.array([1,2,3,4])
print(np.min(var))   # 1
print(np.max(var))   # 4
print(np.sum(var))   # 10
print(np.mean(var))  # 2.5
print(np.argmin(var))   # 0
print(np.argmax(var))   # 3

'''axis = 0 -> column, axis = 1 -> row'''

#Shape and Reshape in Numpy Array

var = np.array([1,2],[1,2])
print(var.shape)    # (2, 2)
print(var.reshape(4,1))    # [[1] [2] [1] [2]]

var1 = np.array([1,2,3,4,5,6],ndim = 4)
print(var1)     # [[[[1] [2]] [[3] [4]] [[5] [6]]]]
print(var1.shape)   # (2, 3, 2, 1)

#Broadcasting in Numpy Array

var = np.array([1,2,3,4])
var1 = np.array([1,2])
print(var + var1)   # [2 4 4 6]

#Indexing Numpy Arrays

var = np.array([1,2,3,4])
print(var[2])   # 3
var1 = np.array([[1,2,3,4],[5,6,7,8]])
print(var1[1,2])    # 7

#Slicing Numpy Arrays

var = np.array([1,2,3,4])
print(var[1:])  # [2 3 4]
var1 = np.array([[1,2,3,4],[5,6,7,8]])
print(var1[1,1:])   # [6 7 8]
print(var1[:,1:])   # [[2 3 4] [6 7 8]]

#Iteration in Numpy Array

var = np.array([1,2,3,4])
for i in var:
    print(i)    # 1 2 3 4

var1 = np.array([[1,2,3,4],[5,6,7,8]])
for i in var1:
    print(i)    # [1 2 3 4] [5 6 7 8]

for i in var1:
    for j in i:
        print(j)    # 1 2 3 4 5 6 7 8

#Copy V/S View

var = np.array([1,2,3,4])
var1 = var.copy()
var1[0] = 10
print(var)  # [1 2 3 4]
print(var1) # [10 2 3 4]

var = np.array([1,2,3,4])
var1 = var.view()
var1[0] = 10
print(var)  # [10 2 3 4]
print(var1) # [10 2 3 4]

'''Copy owns the data, view does not own data
   Copy of an array is new array, view of an array is the same array
   Changes made in copy will not reflect in view and vice versa'''

#Join Array

var1 = np.array([1,2,3,4])
var2 = np.array([5,6,7,8])
var3 = np.concatenate((var1,var2))
print(var3) # [1 2 3 4 5 6 7 8]

var1 = np.array([[1,2],[3,4]])
var2 = np.array([[5,6],[7,8]])
var3 = np.concatenate((var1,var2),axis = 0)
print(var3) # [[1 2] [3 4] [5 6] [7 8]]

#Split Array

var = np.array([1,2,3,4,5,6])
var1 = np.array_split(var,3)
print(var1) # [array([1, 2]), array([3, 4]), array([5, 6])]

#Search Array

var = np.array([1,2,3,4,2,5,2,5,6,7])
print(np.where(var == 2))   # (array([1, 4, 6]),)

#Search Sorted Array

var = np.array([1,2,3,4,5,6,7,8])
print(np.searchsorted(var,2))   # 1

#Sort Array

var = np.array([1,5,6,2,3,4,8,7])
print(np.sort(var)) # [1 2 3 4 5 6 7 8]

#Filter Array

var = np.array([1,2,3,4,5,6,7,8])
print(var[var % 2 == 0])    # [2 4 6 8]

#Shuffle Array

var = np.array([1,2,3,4,5,6,7,8])
np.random.shuffle(var)
print(var)  # [2 4 3 6 1 5 8 7]

#Unique data in Numpy Array

var = np.array([1,2,3,2,5,6,6,8])
print(np.unique(var))    # [1 2 3 5 6 8]

#Resize Array

var = np.array([1,2,3,4,5,6,7,8])
print(var.resize(4,2))  # [[1 2] [3 4] [5 6] [7 8]]

#Insert in Array

var = np.array([1,2,3,4,5,6,7,8])
np.insert(var,2,10)    # [1 2 10 3 4 5 6 7 8]

#Delete in Array

var = np.array([1,2,3,4,5,6,7,8])
np.delete(var,2)    # [1 2 4 5 6 7 8]

#Append in Array

var = np.array([1,2,3,4,5,6,7,8])
np.append(var,10)    # [ 1  2  3  4  5  6  7  8 10]

#Matrix in Numpy

var = np.array([[1,2,3,4],[5,6,7,8]])
print(var)  # [[1 2 3 4] [5 6 7 8]]
print(type(var))    # <class 'numpy.ndarray'>

#Multiplication of Matrix

var1 = np.array([[1,2,3,4],[5,6,7,8]])
var2 = np.array([[1,2,3,4],[5,6,7,8]])
print(var1 * var2)  # [[ 1  4  9 16] [25 36 49 64]]

#Transpose of Matrix

var = np.array([[1,2,3,4],[5,6,7,8]])
print(var.T)    # [[1 5] [2 6] [3 7] [4 8]]

#Dot Product of Matrix

var1 = np.array([[1,2,3,4],[5,6,7,8]])
var2 = np.array([[1,2,3,4],[5,6,7,8]])
print(var1.dot(var2))   # [[ 30  70] [ 70  174]]

#Sum of Matrix

var = np.array([[1,2,3,4],[5,6,7,8]])
print(np.sum(var))  # 36

#Mean of Matrix

var = np.array([[1,2,3,4],[5,6,7,8]])
print(np.mean(var)) # 4.5

#Standard Deviation of Matrix

var = np.array([[1,2,3,4],[5,6,7,8]])
print(np.std(var))  # 2.8722813232690143

#Variance of Matrix

var = np.array([[1,2,3,4],[5,6,7,8]])
print(np.var(var))  # 6.25

#Median of Matrix

var = np.array([[1,2,3,4],[5,6,7,8]])
print(np.median(var))   # 4.5

#Maximum of Matrix

var = np.array([[1,2,3,4],[5,6,7,8]])
print(np.max(var))  # 8

#Minimum of Matrix

var = np.array([[1,2,3,4],[5,6,7,8]])
print(np.min(var))  # 1

#Sort of Matrix

var = np.array([[1,2,3,4],[5,6,7,8]])
print(np.sort(var)) # [[1 2 3 4] [5 6 7 8]]

#Index of Matrix

var = np.array([[1,2,3,4],[5,6,7,8]])
print(np.argmax(var))   # 7
print(np.argmin(var))   # 0

#Flatten of Matrix

var = np.array([[1,2,3,4],[5,6,7,8]])
print(var.flatten())    # [1 2 3 4 5 6 7 8]

#*********************END**********************#