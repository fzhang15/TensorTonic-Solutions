import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        self.word_to_id = {w: id for id, w in enumerate([self.pad_token, self.unk_token, self.bos_token, self.eos_token])}
        words = []
        for text in texts:
            words.extend(text.lower().split())
        unique_words = sorted(list(set(words)))
        for id, w in enumerate(unique_words):
            self.word_to_id[w] = id + 4
        for w, id in self.word_to_id.items():
            self.id_to_word[id] = w
        self.vocab_size = len(unique_words) + 4
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        encoded = []
        if len(text) == 0:
            return encoded
        text = text.lower().split()
            
        for w in text:
            if w in self.word_to_id:
                encoded.append(self.word_to_id[w])
            else:
                encoded.append(self.word_to_id[self.unk_token])
        return encoded
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        decoded = []
        for id in ids:
            decoded.append(self.id_to_word.get(id, self.unk_token))

        return " ".join(decoded)
