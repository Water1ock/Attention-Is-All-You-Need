# Attention-Is-All-You-Need
An implementation of the Transformer Model defined in the "Attention Is All You Need" paper using PyTorch.

![image](https://github.com/user-attachments/assets/d1a869e4-ebac-4ea4-9126-4f2584009427)


# Implementation

Different parts of the transformer model have been implemented in different files of the model_components package.

positional_encoding.py

![image](https://github.com/user-attachments/assets/378d58d8-f560-49c6-8415-f4800cf219ab)


multi_head_attention.py

![image](https://github.com/user-attachments/assets/0931a498-df4b-458e-81af-900df109e8ce)


add_and_norm.py

![image](https://github.com/user-attachments/assets/2661d8c2-7b7c-4265-8236-eb41aa95e291)


feed_forward.py

![image](https://github.com/user-attachments/assets/01d209c4-3761-49d0-9acb-6893c80df19b)


encoder.py and decoder.py

![image](https://github.com/user-attachments/assets/0625618a-ad83-449a-9a43-eb5796ddbf07)

# Configurations

Configurations have been defined in config.py A few have been kept the same as the paper, such as d_model = 512

# TO-DO

1. Training the model.
2. Evaluating the BLEU score.
