import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm


class BPTokenizer():
        def __init__(self, vocab_size, text):
            self.vocab_size = vocab_size
            tokens = text.encode('UTF-8')
            tokens = list(map(int,tokens))
            self.tokens = self.merge_all_reps(tokens)
            
            
        def get_stats(self,tokens):
            # Get the counts of number of repetitions...
            # for each byte pair in the given text 
            counter = {}
            for pair in zip(self.tokens, self.tokens[1:]):
                counter[pair] = counter.get(pair, 0) + 1
            return counter
            
            
        def merge_reps(self, all_tokens, pair_to_merge, new_token):
            # Merge the given pair and create a new id for it.
            
            new_all_tokens = []
            i = 0
            while i < len(all_tokens):
                if i < len(all_tokens) - 1 and all_tokens[i] == pair_to_merge[0] and all_tokens[i+1] == pair_to_merge[1]:
                    new_all_tokens.append(new_token)
                    i += 2
                else:
                    new_all_tokens.append(all_tokens[i])
                    i += 1
            return new_all_tokens
                

        def merge_all_reps(self, all_tokens):
            # Merge all the repeating pairs iteratively with new ids
            
            new_token_count = self.vocab_size - 256
            new_tokens = list(all_tokens)
            self.merges = {}
            for i in tqdm(range(new_token_count)):
                new_id = 256 + i
                counts = self.get_stats(new_tokens)
                top_pair = max(counts, key=counts.get)
                self.merges[top_pair] = new_id
                new_tokens = self.merge_reps(new_tokens, top_pair, new_id)
            return new_tokens
            
            
        def createVocab(self):
            self.vocab = {idx: bytes([idx]) for idx in range(256)}

            # This assumes the sequence in which the merges were created ...
            # is preserved in the dictionary (True in python > 3.7) 
            for (p0,p1), idx in self.merges.items():
                self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        
        
        def decoder(self, ids):
            # Convert Ids to text for inference
            self.createVocab()
            tokens = b''.join(self.vocab[id_] for id_ in ids)
            return tokens.decode("utf-8", errors = 'replace')


        def encoder(self, text):
            # Convert text to ids for LLM training           
            ids = text.encode("UTF-8")
            ids = list(map(int,ids))
            while len(text) > 2:
                stats = self.get_stats(ids)
                pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
                if pair not in self.merges:
                    break #Nothing can be merged
                    
                idx = self.merges[pair]
                ids = self.merge(ids, pair, idx)
            return ids
            


def scrape_url(url, start_pattern = 0, end_pattern = 0):
    
    #Fetch the html content
    response = requests.get(url)
    html_content = response.text
    
    #Parse the html content
    soup = BeautifulSoup(html_content, "html.parser")
    
    #Extract all text from soup
    text = soup.get_text()
    
    return text 
    
    
if __name__ == "__main__":
    url = "https://www.reedbeta.com/blog/programmers-intro-to-unicode/"
    text = scrape_url(url)
    tokenizer = BPTokenizer(vocab_size = 300, text = text)
    