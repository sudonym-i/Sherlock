from pygments.lexers import CLexer
from pygments.token import Token
from pygments import highlight
from pygments.formatters import get_formatter_by_name
import torch
from typing import List, Tuple, Dict

def tokens_to_vectors(tokens_obj: List[str]):
    # right now there is no information inherent in teh vector space,
    # however, each token does ger a unique set ofo coordinates in various
    # vector spaces - dimensions are variable
    vectors = []  # Initialize a list to hold vectors for each token
    vector = []
    for tokens in tokens_obj:

        vector = []  # Reset vector for each token
        for i in range(10):
            vector.append(0)  # Initialize with zeros for each dimension
        i = 0
        for char in tokens:
            vector[i] = ord(char)  # Convert each token to a list of ASCII values
            i += 1
        vectors.append(vector)

        # outputs a list of 10 dimensional vectors = one vector per token
    return vectors


def tokenize_c(c_code):
    
    lexer = CLexer()
    tokens = []
    token_types = []
    
    for token_type, value in lexer.get_tokens(c_code):
        # Filter out whitespace and newlines
        if token_type not in (Token.Text, Token.Text.Whitespace):
            if value.strip():  # Skip empty tokens
                tokens.append(value.strip())
                token_types.append(str(token_type))
    # Return tokens and their types
    return tokens
