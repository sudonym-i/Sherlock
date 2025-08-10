import torch
from torch.utils.data import Dataset, DataLoader
import os
import csv
from text_to_tensor import tokenize_c, tokens_to_vectors


def get_csv_line(filename, line_number):
    with open(filename, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for index, row in enumerate(csv_reader):
            if index == line_number - 1:  # Adjust for 0-based indexing
                return row
    return None  # Line not found



# need to implement datalouder and bring data from dataset without overloading memory
class dataset(Dataset):
    def __init__(self, data_index_dir):
        self.data_dir = data_index_dir

    def __len__(self):
        return len(os.listdir(self.data_dir))
    
    def __getitem__(self, index_line):
        get_csv_line(self.data_dir, index_line)



    def tokenize_c(self, code):
        return tokenize_c(code)
    
    def tokens_to_vectors(self, tokens):
        return tokens_to_vectors(tokens)
    
    def to_tensor(self, vectors):
        torch.tensor(vectors)

