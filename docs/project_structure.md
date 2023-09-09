# PyGenesis Project Structure

## Proposed Structure Guideline

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

### Detailed Structure Recommendation

1. **layers**: Consider renaming the `model` package to `layers` and store all
   layer-related classes/modules here.

   - **dense.py**: For dense layers
   - **convolutional.py**: For CNN layers
   - **recurrent.py**: For RNN layers

2. **errors**: Break up error modules as follows:

   - **squared.py**: Contains squared error loss classes like MSE.
   - **cross_entropy.py**: Contains cross-entropy loss classes.

3. **activations**: Separate activation functions for clarity:

   - **relu.py**: ReLU activation
   - Add more as needed.

4. **utils**: Use this directory for utility functions, data loaders, and other
   helper code.

5. **train.py**, **run.py**, **xor.py**: Consider placing scripts for running
   experiments and training in a separate directory. This keeps them distinct
   from your core library.
