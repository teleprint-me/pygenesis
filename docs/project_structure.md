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

## Revised Project Structure Guideline

Taking into account future developments such as implementing transformers and
techniques like LoRA, QLoRA, and RoPE, consider building a structure that's both
modular and extensible.

```sh
pygenesis
├── __init__.py
├── layers
│   ├── __init__.py
│   ├── base.py  # Base layer interface
│   ├── dense.py
│   ├── convolutional.py
│   ├── recurrent.py
│   └── transformer.py  # new
├── embeddings
│   ├── __init__.py
│   └── positional.py  # new (For RoPE)
├── errors
│   ├── __init__.py
│   ├── squared.py
│   └── cross_entropy.py
├── activations
│   ├── __init__.py
│   └── relu.py
├── regularizers  # new
│   ├── __init__.py
│   └── l2.py
├── techniques  # new (For advanced techniques)
│   ├── __init__.py
│   ├── lora.py  # new
│   ├── qlora.py  # new
│   └── rope.py  # new
├── utils
│   └── __init__.py
├── run.py
├── train.py
└── xor.py
```

### Structural Considerations

1. **layers/transformer.py**: Use this module for implementing the transformer
   architecture.

2. **embeddings/positional.py**: For positional embeddings, including features
   like rotary positional embeddings for RoPE.

3. **regularizers**: Create a separate module for various regularization
   techniques, ensuring modularity and flexibility.

4. **techniques**: A directory dedicated to advanced techniques like LoRA,
   QLoRA, and RoPE.

5. **layers/base.py**: Consider establishing a base class for all layers to
   maintain a consistent API.

### Advanced Techniques

For the implementation of advanced techniques like LoRA and QLoRA, it's
recommended to create abstract base classes or interfaces that define the
expected methods and attributes. This allows for easy integration with existing
layers or models as you expand your project.
