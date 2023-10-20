#%%

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import pandas as pd


#Same function as Task 1, but this time stopwords are defaulted to true.
def preprocessing(passage,stop_words=True): #Easiest/quickest way to toggle the removal of stopwords for the rest of the file is by using this line.
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
#Create a dataframe for the tsv file - using pandas as it uses vectorised operations, e.g. .apply / .map - in the Q&A it was stated we'd need to use vectorised operations and parrallelisation
all_passages = pd.read_csv('candidate-passages-top1000.tsv', sep='\t', header=None, names=['query_id', 'passage_id', 'query', 'passage'])

#Read the test_queries file in the same way.
test_queries = pd.read_csv("test-queries.tsv", sep='\t', header=None, names=["query_id","query"])

#Create a dict for the inverted index
inv_index = {}

unique_passages = all_passages.drop_duplicates(['passage_id'])
avg_passage_len = 0 #This will be used later for BM25

for i, row in unique_passages.iterrows(): #No need to repeat the same passages
    query_id = row['query_id']
    passage_id = row['passage_id']
    queries = preprocessing(row['query']) #choosing to remove stopwords for now
    tokens = preprocessing(row['passage'])
    token_frequency = Counter(tokens) 
    query_token_frequency = Counter(queries)
    words_passage = len(token_frequency) 
    words_query = len(query_token_frequency) 
    avg_passage_len += words_passage #This will be used later for BM25
    for token, freq in token_frequency.items():
        if freq ==1: #Removing any token that only has a frequency of one as they're most likely to be an error/emoji and wont provide much extra info
            del(token)
        else:
            inv_index.setdefault(token, [])
            inv_index[token].append((passage_id, freq, words_passage))


vocabulary = list(inv_index.keys())
values = inv_index.values()


###TF-IDF: TF = number of times term t appears in a document / total number of terms in that document
# IDF = log10(total number of documents / number of documents with term t in it)
N = len(unique_passages) 

tfidf = {}
for token, values in inv_index.items():
    nt = len(values)
    for pid, freq, words_passage in values:
        tee_eff = freq / words_passage
        idf = np.log10(N/nt)
        tfidf[pid, token] = np.clip(np.nan_to_num(tee_eff * idf), a_min=0, a_max=1) #Ensuring if a value of nan appears that it is set to 0 instead. And clipped between 0 and 1 just to be safe.



##### COSINE SIMILARITY & BM25

#Create a function to make a query vector, another to make a passage vector
# then we can use both of those to calculate the cosine similarity

#Cosine similarity: numerator = dot product between two vectors
#Denominator = length of vector x * length of vector y --> sqrt(sum of squared elements of x or y)

# N = len(unique_passages) #Already defined earlier
avgdl = avg_passage_len/N #Used for BM25
k1, k2, b = 1.2, 100, 0.75 #Used for BM25

#TF-IDF vector representation of the passages
def p_vector(tokens,pid):
    p_vector = np.zeros((len(vocabulary)))
    for token in np.unique(tokens):
        try:
            index_of_token = vocabulary.index(token)
            p_vector[index_of_token] = tfidf[(pid,token)]
        except:
            pass
    return p_vector

#Extract the TF-IDF representation of the queries
def q_vector(tokens):
    query_length = len(tokens)
    query_vect = np.zeros((len(vocabulary)))
    query_token_frequency = Counter(tokens)
    for token in np.unique(tokens):
        tf = query_token_frequency[token]/query_length
        try:
            df = len(inv_index[token]) #df = document frequency
            idf = np.log10(N/df)
            token_index = vocabulary.index(token)
            query_vect[token_index] = np.clip(np.nan_to_num(tf * idf), a_min=0, a_max=1)
        except:
            pass
    return query_vect

def BM25(query_tokens, passage_tokens, pid, k1, k2, b):
    BM25score = 0
    qfrequency = Counter(query_tokens)
    frequency = Counter(passage_tokens)
    for token in query_tokens:
        try:
            ni = len(inv_index[token])
        except:
            ni = 0 #If there's a spelling error or a term that doesnt appear in our inv_index it should be given a score of 0. Ideally, we would correct the spelling error.
        
        #term frequency in passage (fi), term frequency in query(qfi), passage length(dl). 
        fi, qfi, dl = frequency[token], qfrequency[token], len(passage_tokens) 
        K = k1 * ((1-b) + b * (dl / avgdl))
        term1 = np.log( (N - ni + 0.5 ) / (ni + 0.5) ) #Easier to split into three terms to avoid making mistakes with brackets etc for the complicated formula
        term2 = ((k1 + 1) * fi) / (K + fi)
        term3 = ((k2 + 1) * qfi) / (k2 * qfi)
        BM25score += term1 * term2 * term3
    return BM25score

#Combined by cosine similarity and BM25 functions so we only have to iterate through the passages once, saving time.
def cosine_similarity_BM25(query, passage, pid):
    """This function computes both the cosine similarity and the BM25 scores for a given passage.
    It computes both in the same function to avoid having to iterate through the rows more than once as
    iterating through the rows is not the most efficient way, so this helps to improve the efficient despite that.
    The k1, k2 and b values are given at the start of this section. If needed, these could be added to the inputs
    for this function.
    Inputs: 
    query = the raw query given
    passage = the current passage associated with the query
    pid = the passage id of the passage above.
    Outputs:
    score = the cosine similarity score for the query and passage
    BM25 score = the BM25 score for the query and passage."""
    query_tokens = preprocessing(query)
    passage_tokens = preprocessing(passage)
    query_vector = q_vector(query_tokens)
    passage_vector = p_vector(passage_tokens, pid)
    BM25score = BM25(query_tokens, passage_tokens, pid, k1, k2, b)
    score = np.dot(query_vector, passage_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(passage_vector))   
    return score, BM25score




#initiate array that will be used for the final qid,pid,score output
qid_pid_scores = np.zeros(3)
qid_pid_BM25scores = np.zeros(3)

#For each row in the test queries document (if we iterate through by the rows it'll maintain the order in the output - as required)
for i, row in test_queries.iterrows():
    curr_qid_pid_scores = np.zeros(3) #Initiate a temporary array for what will be the top 100 scoring pairs for each qid
    curr_qid_pid_BM25scores = np.zeros(3) #Copied for BM25
    qid = row['query_id']
    qid_passages = all_passages[all_passages['query_id'] == qid] #find all the related pid's, passages etc, related to the current qid
    
    #Now we have all the passages that have the same qid lets iterate through them to calculate the cosine sim/BM25 scores before ranking them.
    for i, qrow in qid_passages.iterrows(): 
        query = qrow['query']
        passage = qrow['passage']
        pid = qrow['passage_id']
        score, BM25score = cosine_similarity_BM25(query,passage,pid) #Combined by cosine similarity and BM25 functions so we only have to iterate through the passages once, saving time.
        score_stack = np.array([[qid,pid,score]]) #Create the qid,pid,score row for the pid in question
        curr_qid_pid_scores = np.vstack((curr_qid_pid_scores,score_stack)) #add it to the others for this qid
        
        BM25score_stack = np.array([[qid,pid,BM25score]]) #Repeat for BM25
        curr_qid_pid_BM25scores = np.vstack((curr_qid_pid_BM25scores,BM25score_stack)) 

    curr_qid_pid_scores = curr_qid_pid_scores[1:,:] #remove the row of zeros at the top of the temporary array used to initiate the array
    curr_qid_pid_score = curr_qid_pid_scores[np.argsort(-curr_qid_pid_scores[:,-1])] #sort the current qid-pid pairs in ascending order (using the - in front) by their cosine similarity score
    curr_qid_pid_score = curr_qid_pid_score[:100,:] #Take the top 100 pairs for this qid as requested
    qid_pid_scores = np.vstack((qid_pid_scores,curr_qid_pid_score)) #Add them to the overall array for all qid's
    
    curr_qid_pid_BM25scores = curr_qid_pid_BM25scores[1:,:]  #repeat the above for BM25
    curr_qid_pid_BM25score = curr_qid_pid_BM25scores[np.argsort(-curr_qid_pid_BM25scores[:,-1])] 
    curr_qid_pid_BM25score = curr_qid_pid_BM25score[:100,:] 
    qid_pid_BM25scores = np.vstack((qid_pid_BM25scores,curr_qid_pid_BM25score))

qid_pid_scores = qid_pid_scores[1:,:] #Remove the zeros at the top of the array that we used to initialise it
# print(qid_pid_scores)
np.savetxt("tfidf.csv", qid_pid_scores, fmt='%d,%d,%s', delimiter=',')

qid_pid_BM25scores = qid_pid_BM25scores[1:,:] 
np.savetxt("bm25.csv", qid_pid_BM25scores, fmt='%d,%d,%s', delimiter=',')
        

