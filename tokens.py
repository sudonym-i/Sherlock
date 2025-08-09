from pygments.lexers import CLexer
from pygments.token import Token
from pygments import highlight
from pygments.formatters import get_formatter_by_name

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
                print(value.strip())  # Debugging output
    
    print("Tokens:", tokens[:10])  # First 10 tokens
    print("Types:", token_types[:10])
    return tokens, token_types

