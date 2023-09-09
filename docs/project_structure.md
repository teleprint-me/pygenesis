# PyGenesis Project Structure

---

### Proposed Structure

The main idea is to categorize by functionality and keep related modules
together.

```sh
pygenesis
├── __init__.py
├── layers  # new
│   ├── __init__.py
│   ├── dense.py  # new
│   ├── convolutional.py  # new
│   └── recurrent.py  # new
├── errors  # new
│   ├── __init__.py
│   ├── squared.py  # new
│   └── cross_entropy.py  # new
├── activations  # new
│   ├── __init__.py
│   └── relu.py  # new
├── utils  # new
│   └── __init__.py
├── run.py
├── train.py
└── xor.py
```

### Detailed Explanation

1. **layers**: Instead of having a `model` package, you could rename it to
   `layers` and store all layer-related classes/modules here.

   - **dense.py**: For dense layers
   - **convolutional.py**: For CNN layers
   - **recurrent.py**: For RNN layers

2. **errors**: You could break up your error modules similarly.

   - **squared.py**: Contains squared error loss classes like MSE.
   - **cross_entropy.py**: Contains cross-entropy loss classes.

3. **activations**: It's often beneficial to separate out activation functions.

   - **relu.py**: ReLU activation
   - Add more as needed.

4. **utils**: This could be where you keep utility functions, data loaders, and
   other helper code.

5. **train.py**, **run.py**, **xor.py**: I assume these are for running
   experiments and training. They should likely be separate from your core
   library, possibly even in a separate directory.

---
