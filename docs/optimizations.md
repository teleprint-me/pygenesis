# Optimizations for TMNN

This document outlines various optimization techniques to get the most
performance out of the TMNN library.

## Table of Contents

1. [NumPy with OpenBLAS](#numpy-with-openblas)
2. [Data Types: int8 and float16](#data-types-int8-and-float16)
3. [Coming Soon: Advanced Optimization Techniques](#coming-soon-advanced-optimization-techniques)
4. [Optimization Using Reduced-Precision and Parallelization](#optimization-using-reduced-precision-and-parallelization)
5. [Using Sparse Layers for Optimization](#using-sparse-layers-for-optimization)

---

## NumPy with OpenBLAS

OpenBLAS is an optimized BLAS (Basic Linear Algebra Subprograms) library that
can significantly speed up linear algebra operations, a cornerstone in machine
learning and data science tasks.

### Installation

- Install OpenBLAS from your package manager. For Arch Linux:

  ```sh
  sudo pacman -S openblas
  ```

- Reinstall NumPy to make use of OpenBLAS:
  ```sh
  pip uninstall numpy
  pip install numpy --no-binary numpy  # `numpy` or `:all:`
  ```

### Validation

To ensure NumPy is using OpenBLAS, execute:

```python
import numpy
numpy.__config__.show()
```

Look for `openblas_info` in the output.

```sh
21:36:37 | ~/Documents/code/remote/tmnn
(.venv) git:(main | Δ) λ bpython
bpython version 0.24 on top of Python 3.11.5 ~/Documents/code/remote/tmnn/.venv/bin/python
>>> import numpy
>>> numpy.__config__.show()
blas_armpl_info:
  NOT AVAILABLE
blas_mkl_info:
  NOT AVAILABLE
blas_ssl2_info:
  NOT AVAILABLE
blis_info:
  NOT AVAILABLE
openblas_info:
    libraries = ['openblas', 'openblas']
    library_dirs = ['/usr/lib64']
    language = c
    define_macros = [('HAVE_CBLAS', None)]
blas_opt_info:
    libraries = ['openblas', 'openblas']
    library_dirs = ['/usr/lib64']
    language = c
    define_macros = [('HAVE_CBLAS', None)]
lapack_armpl_info:
  NOT AVAILABLE
lapack_mkl_info:
  NOT AVAILABLE
lapack_ssl2_info:
  NOT AVAILABLE
openblas_lapack_info:
    libraries = ['openblas', 'openblas']
    library_dirs = ['/usr/lib64']
    language = c
    define_macros = [('HAVE_CBLAS', None)]
lapack_opt_info:
    libraries = ['openblas', 'openblas']
    library_dirs = ['/usr/lib64']
    language = c
    define_macros = [('HAVE_CBLAS', None)]
Supported SIMD extensions in this NumPy install:
    baseline = SSE,SSE2,SSE3
    found = SSSE3,SSE41,POPCNT,SSE42,AVX,F16C,FMA3,AVX2,AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL
    not found = AVX512_SPR
```

## Data Types: int8 and float16

Using smaller data types like `int8` for integers and `float16` for
floating-point numbers can reduce memory consumption and may speed up
computations.

```python
import numpy as np

# Using int8
int_array = np.array([1, 2, 3], dtype=np.int8)

# Using float16
float_array = np.array([1.1, 2.2, 3.3], dtype=np.float16)
```

---

## Coming Soon: Advanced Optimization Techniques

### Cython

Cython is essentially Python with C data types. It converts Python code to C and
makes Python code run as fast as native C code. It's particularly effective for
loops and mathematical computations.

#### Installation

```sh
pip install cython
```

#### Usage Example

Create a `.pyx` file and use the following Cython code to sum an array.

```cython
def sum_array(double[:] arr):
    cdef int i
    cdef double result = 0
    for i in range(arr.shape[0]):
        result += arr[i]
    return result
```

Compile this `.pyx` file to generate C code and a shared library.

#### Integration with TMNN

Discuss how you plan to use Cython in your project, perhaps for the most
performance-critical parts of your neural network library.

---

### JIT Compilation with Numba

Numba translates Python functions to optimized machine code using
industry-standard compilers. It's especially good for numerical functions.

#### Installation

```sh
pip install numba
```

#### Usage Example

```python
from numba import jit

@jit(nopython=True)
def sum_array(arr):
    result = 0
    for i in range(arr.shape[0]):
        result += arr[i]
    return result
```

#### Integration with TMNN

Talk about how Numba could accelerate specific parts of your library, such as
matrix multiplications or activation functions.

---

### Parallelization Techniques

Discuss different parallelization techniques like multi-threading,
multiprocessing, and distributed computing.

#### Thread-based Parallelism

Python's Global Interpreter Lock (GIL) is often a bottleneck. However, for
CPU-bound tasks that spend much of their time waiting for external resources,
threading could be beneficial.

#### Process-based Parallelism

Use Python's `multiprocessing` library to create parallel processes. This method
is usually better for CPU-bound tasks.

#### Distributed Computing

Discuss potential distributed computing options, like using Message Passing
Interface (MPI) for Python (mpi4py).

## Optimization Using Reduced-Precision and Parallelization

### Objective

To reduce both CPU and memory usage by applying reduced-precision arithmetic and
parallel computation techniques.

### Memory Optimization

#### 16-bit Floats

NumPy supports 16-bit float types, which can save memory compared to the default
64-bit floats. You can specify this when creating arrays.

```python
import numpy as np
a = np.array([1, 2, 3], dtype=np.float16)
```

#### 4-bit Masking

Further reduce memory requirements by applying a 4-bit mask to your 16-bit
floats.

```python
mask = 0xF0  # Keep only the most significant 4 bits
masked_a = (a.view(np.uint16) & mask).view(np.float16)
```

### Parallelization

Utilize threading or multiprocessing to perform calculations on different parts
of your data.

#### Thread-based Parallelism

```python
from concurrent.futures import ThreadPoolExecutor

def process_chunk(chunk):
    # Do computation here
    return result

with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_chunk, data_chunks))
```

#### Process-based Parallelism

```python
from multiprocessing import Pool

with Pool(processes=4) as pool:
    results = pool.map(process_chunk, data_chunks)
```

### Trade-offs and Risks

- **Loss of Precision**: Reduced-precision arithmetic can affect numerical
  stability.
- **Increased CPU Usage**: Additional bitwise operations for masking might
  increase CPU usage.
- **Code Complexity**: The additional logic adds complexity to the codebase.

## Using Sparse Layers for Optimization

### Objective

To reduce computational complexity and memory footprint by using sparse layers
instead of dense layers where applicable.

### What Are Sparse Layers?

Sparse layers are variations of dense layers in which only a subset of the
neurons are activated. Because many of the connections are zero, sparse layers
require less memory and fewer computational resources.

### When to Use Sparse Layers

- **High Sparsity**: When the input data or activations have a lot of zeros.
- **Memory Constraints**: When there's limited memory available for the model.
- **Computational Constraints**: When you're aiming for faster model inference.

### How to Implement Sparse Layers in Python

You can implement sparse layers manually by using masked operations, or you can
take advantage of existing libraries that support sparse operations.

```python
# Manual Implementation
import numpy as np

def sparse_layer(input, weights, mask, bias):
    masked_weights = np.multiply(weights, mask)
    return np.dot(input, masked_weights) + bias
```

### Trade-offs and Risks

- **Loss of Expressiveness**: Sparse layers might not capture complex patterns
  as effectively as dense layers.
- **Code Complexity**: Requires additional logic for masking and specialized
  backpropagation algorithms.
