from text_to_tensor import tokenize_c, tokens_to_vectors
import numpy as np
import torch
from define import NueralNet

def main():

    tokens = tokenize_c(code)
    vectors = tokens_to_vectors(tokens)
    tensor = torch.tensor(vectors)
    print(np.array(tensor))

    model = NueralNet(10)  

main()