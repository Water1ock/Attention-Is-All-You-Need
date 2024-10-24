# Attention-Is-All-You-Need

An implementation of the Transformer Model defined in the "[Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)" paper using PyTorch.
The transformer model has been trained on the "english"-"italian" subset of the Opus Books dataset loaded through the huggingFace library.

# Implementation

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/user-attachments/assets/1c7179bf-bcb5-493a-b1bb-20e58f6bc79f" alt="image" width="450" style="display: block; margin: auto;" />

The different components of the model have been implemented in the model_components package. The transformer model has been defined in model.py and makes use of the different components from model_components.

1. [model.py](https://github.com/Water1ock/Attention-Is-All-You-Need/blob/main/model.py) -> Implementation of the Transformer model, along with the build_transformer function which is used later for initializing and training the model.
2. [input_embedding.py](https://github.com/Water1ock/Attention-Is-All-You-Need/blob/main/model_components/input_embedding.py) -> Implementation of word embeddings, before passing the input to the transformer model.
3. [positional_encoding.py](https://github.com/Water1ock/Attention-Is-All-You-Need/blob/main/model_components/positional_encoding.py) -> Implementation of positional encoding, allowing the model to learn about the positions in a flexible way, removing the need for recurrence, and giving support to the attention mechanism.
4. [multi_head_attention.py](https://github.com/Water1ock/Attention-Is-All-You-Need/blob/main/model_components/multi_head_attention.py) -> Implementation of the Multi-Head Attention block, making use of the self attention mechanism.
5. [add_and_norm.py](https://github.com/Water1ock/Attention-Is-All-You-Need/blob/main/model_components/add_and_norm.py) -> Implementation of Layer Normalization in the transformer model.
6. [feed_forward.py](https://github.com/Water1ock/Attention-Is-All-You-Need/blob/main/model_components/feed_forward.py) -> Implementation of the Feed Forward Layer in the transformer model.
7. [residual_connection.py](https://github.com/Water1ock/Attention-Is-All-You-Need/blob/main/model_components/residual_connection.py) -> Adding residual connections to different parts of the model as can be witnessed in the figure above. This is done to mitigate the issues of the vanishing gradient problem, and for maintaining original feature information, removing the need for recurrence.
8. [projection_layer.py](https://github.com/Water1ock/Attention-Is-All-You-Need/blob/main/model_components/projection_layer.py) -> Used to map the output of the final layer (usually the decoder layer) to the vocabulary size for generating predictions (e.g., for token classification or next-token prediction). This layer transforms the hidden state representation into logits corresponding to the vocabulary.
9. [encoder.py](https://github.com/Water1ock/Attention-Is-All-You-Need/blob/main/model_components/encoder.py) -> This part of the model processes the input sequence and generates a set of continuous representations (encodings) that capture the contextual information of the input tokens.
10. [decoder.py](https://github.com/Water1ock/Attention-Is-All-You-Need/blob/main/model_components/decoder.py) -> This part of the model generates the output sequence by attending to both the encoder's output and the previously generated tokens in the target sequence.


# Dataset

The [Opus-Books Dataset](https://huggingface.co/datasets/Helsinki-NLP/opus_books) has been used for training the model for 20 epochs. The configurations of the experiment have been defined in the [config.py](https://github.com/Water1ock/Attention-Is-All-You-Need/blob/main/config.py) file. 

# Results

Due to computational limitations, the model was trained for 7 epochs instead of the standard 20, achieving a loss value of 3.42. A basic translation of yes has been attached below. The weights have not been uploaded on the repository due to their large sizes.


![Screenshot 2024-10-24 165502](https://github.com/user-attachments/assets/6e2d1924-5848-4eb9-8185-c3f0f6257ad0)
