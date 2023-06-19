# Instructor Lite
A one-file lightweight reimplementation of the InstructOR model developed by [Su et al. (2023)](https://arxiv.org/abs/2212.09741), with a focus on simplicity and extendability. It depends only on PyTorch and 🤗Transformers. I created this because I wanted to do inference with Instructor models, but I did not want to incur the overhead of installing SentenceTransformers and the custom version of SentenceTransformers required by [the original implementation](https://github.com/HKUNLP/instructor-embedding).

## Limitations
As a result of the focus on simplicity, this implementation has only been tested on inference with pre-trained Instructor checkpoints. It may not work for training or fine-tuning. If you intend to train your own Instructor or replicate the results in the paper, I recommend that you use the original implementation (linked above). But if you just want to perform inference using existing checkpoints, read on.

## Installation
This implementation is contained in a [single file](./instructor.py), and thus it can be used by simply copying the file into your project. 

Steps to install:

1. Ensure your python environment has `torch` and `transformers` installed.
2. Copy `instructor.py` from this repository into your project.

## Usage
To initialize the model, provide a name or path of a model checkpoint. To use one of the pretrained checkpoints from Su et al. you can provide one of the following names from 🤗Hub:

 - [hkunlp/instructor-base](https://huggingface.co/hkunlp/instructor-base)
 - [hkunlp/instructor-large](https://huggingface.co/hkunlp/instructor-large)
 - [hkunlp/instructor-xl](https://huggingface.co/hkunlp/instructor-xl)

The model also accepts an optional `device` parameter which can be used to specify which `torch.device` the model should run on (default is `"cpu"`).  

Once the model is initialized, you can call `model.encode()` on your instruction / text pairs to obtain an encoding object. Calling the model on the encoding object will return both the embedding for the full text as well as per-token embeddings for the instructions and texts. It also accepts a `normalize` parameter, which will normalize the resulting embeddings. 

The encoding object also has a `text_char_to_token()` method, which you can use to extract embeddings for specific tokens from the original text, without worrying about the instruction.

```python
import torch
from instructor import InstructorModel

model = InstructorModel("hkunlp/instructor-base", device=torch.device("cpu"))

pairs = [
    ("Represent the quote for clustering",  "It was the best of times, it was the worst of times..."),
    ("Represent the sentence for retrieval",  "The quick brown fox jumps over the lazy dog."),
]

encoding = model.encode(pairs)

full_text_embs, token_embs = model(encoding, normalize=True)

# an aggregate embedding that represents "It was the best of times, it was the worst of times..." based on the instruction "Represent the quote for clustering"
full_text_embs[0]

# embeddings for each token in the concatenation of the first instruction / text pair:
# "Represent the quote for clusteringIt was the best of times, it was the worst of times..."
token_embs[0] 

# embeddings for the first token in the first text: "It was the best of times, it was the worst of times..."
token_embs[0, encoding.text_char_to_token(0, 0)]

# embeddings for the first token in the second text: "The quick brown fox jumps over the lazy dog."
token_embs[1, encoding.text_char_to_token(1, 0)]
```
