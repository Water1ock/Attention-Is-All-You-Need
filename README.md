# Attention-Is-All-You-Need
An implementation of the Transformer Model defined in the "Attention Is All You Need" paper using PyTorch.

![image](https://github.com/user-attachments/assets/6d468bfb-aa7e-4a43-93b5-b9a66ea25ea2)

The different layers of the model have been implemented separately in the model_components package for better readability and understanding.
add_and_norm.py -> Implementation of the LayerNormalization step.
input_embedding.py -> Implementation of the initial word embeddings before the positional encoding, and subsequent encoder steps. (Embedding of size (sequence_length, d_model = 512))


Further work includes training the model on a dataset for machine translation tasks, for e.g. the Opus Books Dataset.
