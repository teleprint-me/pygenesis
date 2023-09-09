# tmnn Project Structure

## Proposed Structure Guideline

The main idea is to categorize by functionality and keep related modules
together.

```sh
tmnn
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
tmnn
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

## Extended Revised Project Structure Guideline

```sh
tmnn
├── __init__.py
├── layers
│   ├── __init__.py
│   ├── base.py
│   ├── dense.py
│   ├── convolutional.py
│   ├── recurrent.py
│   └── transformer.py
├── embeddings
│   ├── __init__.py
│   ├── word.py  # New (For various word embeddings like Word2Vec, GloVe)
│   └── positional.py  # New (For rotary positional embeddings like RoPE)
├── errors
│   ├── __init__.py
│   ├── squared.py
│   └── cross_entropy.py
├── activations
│   ├── __init__.py
│   └── relu.py
├── regularizers
│   ├── __init__.py
│   └── l2.py
├── techniques
│   ├── __init__.py
│   ├── lora.py
│   ├── qlora.py
│   └── rope.py
├── tokenization  # New
│   ├── __init__.py
│   └── tokenizer.py  # New (For handling various tokenization techniques)
├── encoding  # New
│   ├── __init__.py
│   └── encoder.py  # New (For general encoding strategies)
├── decoding  # New
│   ├── __init__.py
│   └── decoder.py  # New (For general decoding strategies)
├── utils
│   └── __init__.py
├── run.py
├── train.py
└── xor.py
```

### Notes on the New Modules:

1. **embeddings/word.py**: This module can handle various types of word
   embeddings like Word2Vec, GloVe, or custom embeddings.

2. **tokenization/tokenizer.py**: Consider including this module for handling
   various tokenization techniques, such as sub-word tokenization.

3. **encoding/encoder.py and decoding/decoder.py**: These modules can contain
   your general encoding and decoding strategies. For example, the encoder could
   handle mapping a sequence of tokens to a sequence of vectors, while the
   decoder could produce an output sequence from these vectors.

4. **regularizers, techniques, layers, etc.**: These modules can be further
   modularized to support various algorithms, particularly useful when
   incorporating advanced techniques like LoRA, QLoRA, and RoPE.

By adopting this modular approach, you can facilitate flexibility in mixing and
matching different components, making it suitable for both research experiments
and real-world applications. This structure ensures that your codebase remains
maintainable and comprehensible as you integrate advanced features into your
project.
