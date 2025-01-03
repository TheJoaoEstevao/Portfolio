from typing import List, Dict, Optional, Union
import regex as re
from pathlib import Path
import json
import numpy as np
from collections import defaultdict

class Tokenizer:
    """Modern tokenizer implementation with byte-level BPE"""
    
    def __init__(self, vocab_size: int = 50257, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.encoder: Dict[str, int] = {}
        self.decoder: Dict[int, str] = {}
        self.bpe_ranks: Dict[tuple, int] = {}
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        # Special tokens
        self.special_tokens = {
            "<|endoftext|>": 0,
            "<|pad|>": 1,
            "<|mask|>": 2,
            "<|begin|>": 3,
            "<|end|>": 4
        }
        
    def byte_encode(self, text: str) -> List[int]:
        """Encode text into bytes."""
        return list(text.encode("utf-8"))
    
    def get_pairs(self, word: List[str]) -> set:
        """Get all pairs of consecutive symbols in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def train(self, texts: List[str], vocab_size: Optional[int] = None) -> None:
        """Train the tokenizer on a list of texts."""
        if vocab_size is not None:
            self.vocab_size = vocab_size
            
        # Count word frequencies
        word_freqs = defaultdict(int)
        for text in texts:
            words = self.pat.findall(text)
            for word in words:
                word_freqs[" " + word] += 1
                
        # Initialize character vocabulary
        chars = set()
        for word in word_freqs:
            chars.update(word)
        
        # Initialize with characters and special tokens
        vocab = list(chars)
        vocab = [c for c in vocab if len(c.encode("utf-8")) == 1]
        vocab = sorted(vocab) + ["</w>"]
        
        for special_token in self.special_tokens:
            self.encoder[special_token] = self.special_tokens[special_token]
            self.decoder[self.special_tokens[special_token]] = special_token
            
        # Train BPE
        merges: Dict[tuple, int] = {}
        vocab_size = len(self.encoder)
        
        while vocab_size < self.vocab_size:
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                if freq < self.min_frequency:
                    continue
                    
                symbols = list(word)
                if len(symbols) == 1:
                    continue
                    
                for pair in self.get_pairs(symbols):
                    pairs[pair] += freq
                    
            if not pairs:
                break
                
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            new_token = "".join(best_pair)
            vocab.append(new_token)
            merges[best_pair] = new_token
            
            # Update vocabulary
            self.encoder[new_token] = vocab_size
            self.decoder[vocab_size] = new_token
            vocab_size += 1
            
            # Update word frequencies with merged pairs
            new_word_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                if freq < self.min_frequency:
                    continue
                    
                new_word = word
                while True:
                    symbols = list(new_word)
                    pairs = self.get_pairs(symbols)
                    if not pairs:
                        break
                        
                    bigram = max(pairs, key=lambda x: merges.get(x, -1))
                    if bigram not in merges:
                        break
                        
                    new_word = new_word.replace("".join(bigram), merges[bigram])
                    
                new_word_freqs[new_word] += freq
                
            word_freqs = new_word_freqs
            
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges.keys())}
        
    def bpe(self, token: str) -> List[str]:
        """Apply Byte-Pair Encoding to a token."""
        if token in self.encoder:
            return [token]
            
        word = list(token)
        pairs = self.get_pairs(word)
        
        if not pairs:
            return [token + "</w>"]
            
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
                
            first, second = bigram
            new_word = []
            i = 0
            
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    if j + 1 < len(word) and word[j + 1] == second:
                        new_word.append(first + second)
                        i = j + 2
                    else:
                        new_word.append(word[j])
                        i = j + 1
                except ValueError:
                    new_word.extend(word[i:])
                    break
                    
            word = new_word
            pairs = self.get_pairs(word)
            if not pairs:
                break
                
        return word
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        bpe_tokens: List[str] = []
        for token in self.pat.findall(text):
            token = "".join(self.byte_encode(token))
            bpe_tokens.extend(self.bpe(token))
            
        return [self.encoder[token] for token in bpe_tokens]
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token ids to text."""
        text = "".join([self.decoder[token] for token in tokens])
        return bytearray([int(b) for b in text.split()]).decode("utf-8", errors="replace")
    
    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "min_frequency": self.min_frequency,
                "encoder": self.encoder,
                "decoder": self.decoder,
                "bpe_ranks": {",".join(k): v for k, v in self.bpe_ranks.items()},
                "special_tokens": self.special_tokens
            }, f, ensure_ascii=False, indent=2)
            
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Tokenizer":
        """Load tokenizer from file."""
        path = Path(path)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        tokenizer = cls(data["vocab_size"], data["min_frequency"])
        tokenizer.encoder = data["encoder"]
        tokenizer.decoder = {int(k): v for k, v in data["decoder"].items()}
        tokenizer.bpe_ranks = {tuple(k.split(",")): v for k, v in data["bpe_ranks"].items()}
        tokenizer.special_tokens = data["special_tokens"]
        
        return tokenizer
    
class TokenizerTrainer:
    """Trainer for the tokenizer with advanced features"""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        min_frequency: int = 2,
        special_tokens: Optional[Dict[str, int]] = None,
        lowercase: bool = False,
        unicode_normalizer: Optional[str] = "NFKC"
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or {}
        self.lowercase = lowercase
        self.unicode_normalizer = unicode_normalizer
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text before tokenization."""
        if self.lowercase:
            text = text.lower()
            
        if self.unicode_normalizer:
            import unicodedata
            text = unicodedata.normalize(self.unicode_normalizer, text)
            
        return text
    
    def train(self, texts: List[str], output_path: Union[str, Path]) -> Tokenizer:
        """Train a new tokenizer on texts."""
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Initialize and train tokenizer
        tokenizer = Tokenizer(self.vocab_size, self.min_frequency)
        if self.special_tokens:
            tokenizer.special_tokens.update(self.special_tokens)
            
        tokenizer.train(processed_texts)
        
        # Save trained tokenizer
        tokenizer.save(output_path)
        return tokenizer
    
    @staticmethod
    def train_from_files(
        file_paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        **kwargs
    ) -> Tokenizer:
        """Train tokenizer from text files."""
        texts = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                texts.extend(f.readlines())
                
        trainer = TokenizerTrainer(**kwargs)
        return trainer.train(texts, output_path) 