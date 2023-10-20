#%%%

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

## DATA SOURCES FOR THIS TASK

with open("passage-collection.txt") as collection:
  sentence = collection.read()


#Define our tokeniser function, with an optional argument for if you want to exclude stopwords
def preprocessing(passage,stop_words=False):
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

tokens = preprocessing(sentence, False)
print("total number of tokens: ", len(tokens))


def uniquetokens(tokens):
    """finds the number of unique tokens given a list of tokens.
    input: list of tokens
    output: list of unique tokens"""
    unique_tokens = list(set(tokens)) #Set only adds unique values, turn the set into a list
    return unique_tokens

unique_tokens = uniquetokens(tokens) 
print("unique tokens: ", len(unique_tokens))


def freqoccur(document, stop_words=False):
    """Counts the number of occurences of terms in the provided data set.
    Input: Document of passages (sentences)
    Output: zipped list of the unique tokens and their corresponding frequency"""

    tokens = preprocessing(document, stop_words)
    unique_tokens = uniquetokens(tokens)
    unique_freq = Counter(tokens) 
    return unique_freq



unique_freq = freqoccur(sentence, True) ##TOGGLE STOPWORDS ON/OFF HERE
ranked_unique_freq = unique_freq.most_common()
#Option to print the raw number of occurences or terms in the provided data set, or the number of occurences in order from most to least.
#Uncomment either line if you want to see either.
#Normalised frequency is calculated below.

# print(unique_freq)
# print(ranked_unique_freq)


words = []
values = []
for i in range(len(ranked_unique_freq)):
    k , j = ranked_unique_freq[i]
    words.append(k)
    values.append(j)


print("word len: ", len(words), sum(values))
words = np.asarray(words)
values = np.asarray(values)

#Normalised frequency
normalised_freq = np.divide(values, sum(values))
print(len(normalised_freq))


#Plot the probability of occurrence vs the frequency ranking as requested
plt.plot(np.arange(1, len(words)+1), normalised_freq)
plt.title("Probability of Occurrence vs. Frequency Ranking")
plt.xlabel("Frequency ranking")
plt.ylabel("Probability of occurrence")
plt.savefig("D1_1.pdf")
plt.show()

zipfs_law = np.divide(1/np.arange(1, len(words)+1), np.sum(1/np.arange(1, len(words)+1)))


#Plot our empirical dist vs Zipfs
plt.loglog(np.arange(1, len(words)+1), normalised_freq, label="Emp dist")
plt.loglog(np.arange(1, len(words)+1), zipfs_law, label="Zipfs dist")
plt.legend(["Observed distribution", "Theoretical distribution (Zipf's law)"])
plt.title("Log-Log Plot: Empirical vs. Zipf's Distribution")
plt.xlabel("Frequency ranking (log)")
plt.ylabel("Probability of occurrence (log)")
plt.savefig("D1_2.pdf")
plt.show()



# %%
