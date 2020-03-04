# python is a high level dynamically typed programming language
# emphasis on powerful ideas in very little code that is human readable
# list comprehensions are very powerful
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))

# working in Python 3

# basic data types
# numbers (integers and floats)
x = 3
print(x, type(x))

print(x + 1) # add
print(x - 1) # subtract
print(x * 2) # multiplication
print(x ** 2) # exponentation

x += 1
x *= 2

y = 2.5
print(type(y))

# python has no ++ or -- operators
# python also supports long/complex numbers

# python implements normal boolean operators as English words
t, f = True, False
print(t and f) # resolves to False
print(t or f) # True
print(not t) # False
print(t != f) # True

# strings can be in either single or double quotes
hello = 'hello'
world = "world"
# concatenated via + signs
hw = hello + '' + world
# can include variables via sprintf style formatting
hwsf = '%s %s %d' % (hello, world, 12)
# can also use f strings in Python 3 to include variables directly into string
print(f"{hello} {world}!")

# number of useful methods
s = "hello"
print(s.capitalize()) # "Hello"
print(s.upper()) # "HELLO"
print(s.rjust(7)) # "       hello"
print(s.center(7)) # " hello "
print(s.replace("l", "(ell)")) # "he(ell)(ell)o"
print("    world    ".strip()) # "world"

# containers
# lists -> python equivalent of the array
# resizable and can contain multiple different objects
xs = [3,1,2]
print(xs, xs[2]) # [3,1,2], 3
# can use negative indices to get elements from the back
print(xs[-1]) # 2
# can contain elements of different types
xs[2] = "foo"
print(xs)
xs.append("bar")
print(xs) # [3,1,"foo", "bar"]
xs.pop()
print(x, xs) # ["bar", [3,1,"foo"]]
# python also offers "slicing" syntax
nums = list(range(5))
print(nums) # [0,1,2,3,4]
print(nums[2:4]) # [2,3]
print(nums[2:]) # [2,3,4]
print(nums[:2]) # [0,1]
print(nums[:]) # [0,1,2,3,4]
print(nums[:-1]) # [0,1,2,3]
nums[2:4] = [8,9]
print(nums) # [0,1,8,9,4]

# loops
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)
# can use enumeration
for idx, animal in enumerate(animals):
    print(f"{idx} : {animal}")

# list comprehensions
# creating lists based on an existing list via evaluating an expression
# squaring all numbers in a list WITHOUT comprehensions:
nums = list(range(5))
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)
# squaring all numbers in a list WITH list comprehensions:
nums = list(range(5))
squares = [x ** 2 for x in nums]
print(squares)
# can also include conditions
nums = list(range(5))
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares) # [0, 4, 16]

# dictionaries
# (key, value) pairs (think Maps in Java or JavaScript objects)
d = {'cat': 'cute', 'dog': 'furry'}
print(d['cat']) # 'cute'
print('cat' in d) # True
# adding to dictionaries
d['fish'] = 'wet'
print(d['fish']) # 'wet'
# throws a KeyError if a key is not in the dictionary
# can specify a default cause to handle a key not being in a dictionary
print(d.get("monkey", "not found!"))
print(d.get("fish", "not found!"))
# can iterate over the keys in a dictionary
d = {"person": 2, "cat": 4, "spider" : 8}
for animal in d:
    legs = d[animal]
    print(f"A(n) {animal} has {legs} legs")
# can also iterate over keys and their values
for animal, legs in d.items():
    print(f"A(n) {animal} has {legs} legs")
# can also do dictionary comprehensions
# like a list comprehension, but for a dictionary
nums = list(range(5))
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)

# sets
# unordered collection of distinct elements
animals = {"cat", "dog"}
print("cat" in animals)
print("fish" in animals)
# can only add an element ONCE to a set
animals.add("fish")
print("fish" in animals)
print(len(animals))
# adding an element again does nothing
animals.add("fish")
print(len(animals))
animals.remove("fish")
print(len(animals))
# same syntax for looping as a list
# lack of order means that you cannot make assumptions how you visit the set
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print(f"{idx + 1} : {animal}")
# set comprehensions are also a thing
from math import sqrt
print({int(sqrt(x)) for x in range(30)}) # set([0,1,2,3,4,5])

# tuples
# immutable ordered list of values
# CAN be used as keys in dictionaries/elements of sets, where lists cannot
d = {(x, x + 1) for x in range(10)}
t = (5,6)
print(type(t))
# print(d[t])
# print(d[(1,2)])

# functions
# defined using def
def sign(x):
    if (x > 0):
        return "positive"
    elif (x < 0):
        return "negative"
    else:
        return "zero"

for x in [-1,0,1]:
    print(sign(x))

# can define parameters for functions to take
def hello(name, loud=False):
    if loud:
        print(f"HELLO, {name.upper()}!")
    else:
        print(f"Hello, {name}!")

# classes
class Greeter:
    # constructor
    def __init__(self, name):
        self.name = name # instance variable

    def greet(self, loud=False):
        if loud:
            print(f"HELLO, {self.name.upper()}!")
        else:
            print(f"Hello, {self.name}!")

g = Greeter('Fred') # constructs Greeter instance
g.greet() # calls instance method
g.greet(loud=True)

# numpy
# core library for scientific computing in Python
# provides high performance multidimensional array objects
import numpy as np
# np arrays
# grid of values of the same type
# indexed by a tuple of non-negative integers
# number of dimensions is the rank of the array
# shape of the array is a tuple of integers giving size for each dimension
a = np.array([1,2,3]) # rank 1 array
print(type(a), a.shape, a[0], a[1], a[2])
a[0] = 5
print(a)

b = np.array([[1,2,3],[4,5,6]])
print(b)

print(b.shape)
print(b[0,0], b[0,1], b[1,0])

a = np.zeros((2,2)) # creates 2 x 2 array populated by zeroes
print(a)

b = np.ones((1,2)) # creates 1 x 2 array populated by ones
print(b)

c = np.full((2,2), 7) # creates 2 x 2 array populated by the given number
print(c)

d = np.eye(2) # creates a 2 x 2 identity matrix (populated along diagonal)
print(d)

e = np.random.random((2,2)) # creates a 2 x 2 array populated by random values
print(e)

# array indexing
# can slice numpy arrays, but need to specify slices for each dimension
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
# slicing to pull out first two rows and columns 1 and 2
# [[2 3]
# [6 7]]
b = a[:2, 1:3]
print(b)
# SLICING IS A VIEW INTO DATA, SO MODIFYING THE SLICE MODIFIES THE ORIGINAL DATA
print(a[0,1]) # 2
b[0,0] = 77
print(a[0,1]) # 77

# can mix and match integer/slice indexing, but result has different rank
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
row_r3 = a[[1], :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)
print(row_r2, row_r2.shape)
print(row_r3, row_r3.shape)

# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)
print(col_r2, col_r2.shape)

# slice indexing results in subarrays
# integer indexing results in new arrays
a = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(a[[0, 1, 2], [0, 1, 0]])

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))

# integer array indexing allows for selecting/mutating a single element in a row
# of a matrix
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print (a)

# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10
print (a)

# boolean indexing: can specify indexes to extract from that meet a condition
a = np.array([[1,2], [3, 4], [5, 6]])
bool_idx = (a > 2)
print (bool_idx)
# can use this boolean index to then construct an array
print(a[bool_idx])
# can also be done as
print(a[a>2])

# datatypes
# all numpy arrays are grids of elements of the same type
# numpy has a number of numeric datatypes
# will attempt to guess the datatype when array is created
# can also be specified in constructor

x = np.array([1,2])
y = np.array([1.0,2.0])
z = np.array([1,2], dtype=np.int64)

print(x.dtype, y.dtype, z.dtype)

# mathematical operations operate elementwise on arrays
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
print (x + y)
print (np.add(x, y))

# * is elementwise multiplication, not matrix multiplication
# dot product is used to computer inner products of vectors
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print (v.dot(w))
print (np.dot(v, w))

# can sum up arrays using sum
# may need to reshape data
x = np.array([[1,2],[3,4]])
print(x)
print(x.T)

# broadcasting can allow you to work with arrays of different shapes
# using smaller arrays to change large arrays without creating duplicates

# matplotlib
# used to plot visuals
import matplotlib.pyplot as plt

# plot allows you to plot 2d data
# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)
plt.show()

y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()

# can display multiple things in the same figure via subplot
# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()
