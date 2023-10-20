import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import pandas as pd



def preprocessing(passage,stop_words=True): #Same function as Task 1, but this time stopwords are defaulted to true.
    """Function that takes sentences as input and preprocesses them to output tokens.
    There is the option to remove stopwords, which the default is set to False in Task 1 because we are asked not to remove them.
    However it is set to True in Task 2 and beyond, as that was the choice I made.
    Input: Passage of sentences/words.
    Output: list of preprocessed tokens"""
    tokenizer = RegexpTokenizer(r'\w+')
    passage = passage.lower()
    tok_pass = tokenizer.tokenize(passage)
    tok_pass = [tok for tok in tok_pass if tok.isalpha()]
    if stop_words == True:
        stop_words = stopwords.words('english')
        tokens = [tok for tok in tok_pass if tok not in stop_words]
    else:
        tokens = tok_pass
    return tokens

## DATA SOURCES FOR THIS TASK

#Create a dataframe for the tsv file - using pandas
all_passages = pd.read_csv('candidate-passages-top1000.tsv', sep='\t', header=None, names=['query_id', 'passage_id', 'query', 'passage'])

#Create a dict for the inverted index
inv_index = {}

unique_passages = all_passages.drop_duplicates(['passage_id'])

for i, row in unique_passages.iterrows(): #No need to repeat the same passages
    passage_id = row['passage_id']
    tokens = preprocessing(row['passage']) #choosing to remove stopwords for now
    token_frequency = Counter(tokens) 
    words_passage = len(token_frequency) 
                                    
    for token, freq in token_frequency.items():
        if freq ==1: #Removing any token that only has a frequency of one as they're most likely to be an error/emoji and wont provide much extra info
            del(token)
        else:
            inv_index.setdefault(token, [])
            inv_index[token].append((passage_id, freq, words_passage))

print(inv_index)
            
#To see the length of the inverted index uncomment the below
# print(len(inv_index))

#If you want to see the first two items of the inverted index, or the vocabulary uncomment the lines below.
# print(dict(list(inv_index.items())[0:2]))
# vocabulary = list(inv_index.keys())