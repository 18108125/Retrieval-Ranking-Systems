#%%

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import pandas as pd

## DATA SOURCES FOR THIS TASK
#Create a dataframe for the tsv file - using pandas
all_passages = pd.read_csv('candidate-passages-top1000.tsv', sep='\t', header=None, names=['query_id', 'passage_id', 'query', 'passage'])

unique_passages = all_passages.drop_duplicates(['passage_id'])

test_queries = pd.read_csv("test-queries.tsv", sep='\t', header=None, names=["query_id","query"])


N = len(unique_passages) 
#Same function as Task 1, but this time stopwords are defaulted to true.
def preprocessing(passage,stop_words=True):#Easiest/quickest way to toggle the removal of stopwords for the rest of the file is by using this line.
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

#Create a dict for the inverted index
def inv_index_func(unique_passages):
    """Creates an inverted index when given a list of passages. Decided to create a function for it in Task 4 so I can minimise it."""
    inv_index = {}
    for i, row in unique_passages.iterrows(): #No need to repeat the same passages
        passage_id = row['passage_id']
        tokens = preprocessing(row['passage'])#choosing to remove stopwords for now
        token_frequency = Counter(tokens) 
        words_passage = len(token_frequency) 
        
        for token, freq in token_frequency.items():
            if freq ==1: #Removing any token that only has a frequency of one as they're most likely to be an error/emoji and wont provide much extra info
                del(token)
            else:
                inv_index.setdefault(token, [])
                inv_index[token].append((passage_id, freq, words_passage))
    return inv_index

inv_index = inv_index_func(unique_passages)

vocabulary = list(inv_index.keys())
values = inv_index.values()

V = len(vocabulary) #number of unique words in the entire collection - used in Laplace.

def Laplace_Lidstone(query, passage, epsilon=0.1):
    """This function takes in a query, passage and a value for epsilon (defauled to 0.1) and outputs both the Laplace and Lidstone scores
    Similarly to Task 3, the decision was taken to compute both in one function to avoid iterating through the data too many times."""
    query_tokens = preprocessing(query)
    passage_tokens = preprocessing(passage)
    Laplacescore = 0
    Lidstonescore = 0
    frequency = Counter(passage_tokens)
    D = len(passage_tokens)

    for token in query_tokens:
        if token not in vocabulary:
            continue
        freq = frequency[token] + 1 #one added for Laplace smoothing
        prob = freq / (D + V)
        Laplacescore += np.log(prob) #Ensured to use np.log rather than np.log10 as it was specified to take the natural logarithm
        lidfreq = frequency[token] + epsilon #epsilon added for Lidstone smoothing
        lidprob = lidfreq / ( (D+epsilon) + V) #epsilon added for Lidstone smoothing
        Lidstonescore += np.log(lidprob) #Ensured to use np.log rather than np.log10 as it was specified to take the natural logarithm
    return Laplacescore, Lidstonescore

#Estimate for words that occur is lamba P(w|D) +  (1-lamba)P(w|C)
#Dirichlet: lambda = N/(N+µ), (1-lamba) = µ/(N+µ). Where N = passage length

def Dirichlet(query, passage, mu=50): #default value of mu = 50 as given in the assignment, but the option to change it is there if needed.
    Dirichletscore = 0
    query_tokens = preprocessing(query)
    passage_tokens = preprocessing(passage)
    frequency = Counter(passage_tokens)
    D = len(passage_tokens)

    for token in query_tokens:
        if token not in vocabulary:
            continue
        
        lambda_param = D / (D + mu) #D = document length
        one_minus_lambda = mu / (D + mu)

        prob_doc = lambda_param * (frequency[token] / D)
        prob_col = one_minus_lambda * ( len(inv_index[token]) / V)

        Dirichletscore += np.log(prob_doc + prob_col) #Ensured to use np.log rather than np.log10 as it was specified to take the natural logarithm
    return Dirichletscore


#initiate array that will be used for the final qid,pid,score output
qid_pid_Laplacescores = np.zeros(3)
qid_pid_Lidstonescores = np.zeros(3)
qid_pid_Dirichletscores = np.zeros(3)

#for each row in the test queries document (if we iterate through by the rows it'll maintain the order)
for i, row in test_queries.iterrows():
    curr_qid_pid_Laplacescores = np.zeros(3) #Initiate a temporary array for what will be the top 100 scoring pairs for each qid
    curr_qid_pid_Lidstonescores = np.zeros(3) #Copied for Lidstone
    curr_qid_pid_Dirichletscores = np.zeros(3) #Copied for Dirichlet
    qid = row['query_id']
    qid_passages = all_passages[all_passages['query_id'] == qid] #find all the related pid's, passages etc, related to the current qid
    for i, qrow in qid_passages.iterrows(): 
        query = qrow['query']
        passage = qrow['passage']
        pid = qrow['passage_id']
        Laplacescore, Lidstonescore = Laplace_Lidstone(query,passage, epsilon=0.1) #Combined by Lidstone and Laplace functions so we only have to iterate through the passages once, saving time.
        Dirichletscore = Dirichlet(query,passage,mu=50)
        
        Laplacescore_stack = np.array([[qid,pid,Laplacescore]]) 
        curr_qid_pid_Laplacescores = np.vstack((curr_qid_pid_Laplacescores,Laplacescore_stack)) #add it to the others for this qid
        Lidstonescore_stack = np.array([[qid,pid,Lidstonescore]]) 
        curr_qid_pid_Lidstonescores = np.vstack((curr_qid_pid_Lidstonescores,Lidstonescore_stack)) 
        Dirichletscore_stack = np.array([[qid,pid,Dirichletscore]]) 
        curr_qid_pid_Dirichletscores = np.vstack((curr_qid_pid_Dirichletscores,Dirichletscore_stack)) 
    
    curr_qid_pid_Laplacescores = curr_qid_pid_Laplacescores[1:,:] #remove the row of zeros at the top of the temporary array
    curr_qid_pid_Laplacescore = curr_qid_pid_Laplacescores[np.argsort(-curr_qid_pid_Laplacescores[:,-1])] #sort the current qid-pid pairs by their Laplace score
    curr_qid_pid_Laplacescore = curr_qid_pid_Laplacescore[:100,:] #for the row above argsort sorts in ascending order so - in front makes it sort in descending order
    qid_pid_Laplacescores = np.vstack((qid_pid_Laplacescores,curr_qid_pid_Laplacescore))
    
    curr_qid_pid_Lidstonescores = curr_qid_pid_Lidstonescores[1:,:] 
    curr_qid_pid_Lidstonescore = curr_qid_pid_Lidstonescores[np.argsort(-curr_qid_pid_Lidstonescores[:,-1])] 
    curr_qid_pid_Lidstonescore = curr_qid_pid_Lidstonescore[:100,:] 
    qid_pid_Lidstonescores = np.vstack((qid_pid_Lidstonescores,curr_qid_pid_Lidstonescore))
    
    curr_qid_pid_Dirichletscores = curr_qid_pid_Dirichletscores[1:,:] 
    curr_qid_pid_Dirichletscore = curr_qid_pid_Dirichletscores[np.argsort(-curr_qid_pid_Dirichletscores[:,-1])] 
    curr_qid_pid_Dirichletscore = curr_qid_pid_Dirichletscore[:100,:] 
    qid_pid_Dirichletscores = np.vstack((qid_pid_Dirichletscores,curr_qid_pid_Dirichletscore))


qid_pid_Laplacescores = qid_pid_Laplacescores[1:,:] #Remove the zeros at the top of the array that we used to initialise it
np.savetxt("laplace.csv", qid_pid_Laplacescores, fmt='%d,%d,%s', delimiter=',')

qid_pid_Lidstonescores = qid_pid_Lidstonescores[1:,:] #Remove the zeros at the top of the array that we used to initialise it
np.savetxt("lidstone.csv", qid_pid_Lidstonescores, fmt='%d,%d,%s', delimiter=',')

qid_pid_Dirichletscores = qid_pid_Dirichletscores[1:,:] #Remove the zeros at the top of the array that we used to initialise it
np.savetxt("dirichlet.csv", qid_pid_Dirichletscores, fmt='%d,%d,%s', delimiter=',')




