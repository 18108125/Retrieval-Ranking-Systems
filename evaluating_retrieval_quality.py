#%%

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import pandas as pd


#Copied from CW1
def preprocessing(passage,stop_words=True): #Easiest/quickest way to toggle the removal of stopwords for the rest of the file is by using this line.
    """Function that takes sentences as input and preprocesses them to output tokens.
    There is the option to remove stopwords, which the default is set to True.

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
# all_passages = pd.read_csv('candidate_passages_top1000.tsv', sep='\t', header=None, names=['query_id', 'passage_id', 'query', 'passage'])
all_passages = pd.read_csv('candidate_passages_top1000.tsv', sep='\t', header=None, names=['query_id', 'passage_id', 'query', 'passage'])


#Read the test_queries file in the same way.
test_queries = pd.read_csv("test-queries.tsv", sep='\t', header=None, names=["query_id","query"])
validation_queries = pd.read_csv("validation_data.tsv", sep='\t', header=0, names=['query_id', 'passage_id', 'query', 'passage', 'relevance'])

unique_passages = all_passages.drop_duplicates(['passage_id'])
unique_validation = validation_queries.drop_duplicates(['passage_id'])
unique_validation_qid = validation_queries.drop_duplicates(['query_id'])
avg_passage_len = 0 #This will be used later for BM25

# ##UNCOMMENT ONCE TASK FINISHED
#Create a dict for the inverted index
inv_index = {}

# for i, row in unique_validation.iterrows(): #No need to repeat the same passages
for i, row in unique_passages.iterrows(): #No need to repeat the same passages
    query_id = row['query_id']
    passage_id = row['passage_id']
    relevance = ['relevance']
    queries = preprocessing(row['query']) #choosing to remove stopwords for now
    tokens = preprocessing(row['passage'])
    token_frequency = Counter(tokens) 
    query_token_frequency = Counter(queries)
    words_passage = len(token_frequency) 
    words_query = len(query_token_frequency) 
    avg_passage_len += words_passage #This will be used later for BM25
    for token, freq in token_frequency.items():
            inv_index.setdefault(token, [])
            inv_index[token].append((passage_id, freq, words_passage, relevance))
        # if freq ==1: #Removing any token that only has a frequency of one as they're most likely to be an error/emoji and wont provide much extra info
        #     del(token)
        # else:
            # inv_index.setdefault(token, [])
            # inv_index[token].append((passage_id, freq, words_passage, relevance))
# ##UNCOMMENT ONCE TASK FINISHED

np.save('inv_index.npy', inv_index)
inv_index = np.load('inv_index.npy', allow_pickle=True).item()
np.save('avg_passage_len.npy', avg_passage_len)
avg_passage_len = np.load('avg_passage_len.npy', allow_pickle=True).item()

vocabulary = list(inv_index.keys())
values = inv_index.values()

###TF-IDF: TF = number of times term t appears in a document / total number of terms in that document
# IDF = log10(total number of documents / number of documents with term t in it)
N = len(unique_passages) 

# ##UNCOMMENT ONCE TASK FINISHED
tfidf = {}
for token, values in inv_index.items():
    nt = len(values)
    for pid, freq, words_passage, relevance in values:
        tee_eff = freq / words_passage
        idf = np.log10(N/nt)
        tfidf[pid, token] = np.clip(np.nan_to_num(tee_eff * idf), a_min=0, a_max=1) #Ensuring if a value of nan appears that it is set to 0 instead. And clipped between 0 and 1 just to be safe.

np.save('tfidf.npy', tfidf)
# ##UNCOMMENT ONCE TASK FINISHED
tfidf = np.load('tfidf.npy', allow_pickle=True).item()

#%% BM25 FUNCTION FORMULATION

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
    # score = np.dot(query_vector, passage_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(passage_vector))   
    return BM25score
    # return score, BM25score


#%% BM25 EXECUTION

#initiate array that will be used for the final qid,pid,score output
# qid_pid_scores = np.zeros(3)

# # ##UNCOMMENT ONCE TASK FINISHED
qid_pid_BM25scores = np.zeros(4)

bm25 = {}
#For each row in the test queries document (if we iterate through by the rows it'll maintain the order in the output - as required)
# for i, row in test_queries.iterrows():
# for i, row in unique_validation_qid.iterrows():
for i, row in unique_validation_qid.iterrows(): #iterate through all the queries
    # curr_qid_pid_scores = np.zeros(3) #Initiate a temporary array for what will be the top 100 scoring pairs for each qid
    curr_qid_pid_BM25scores = np.zeros(4) #Copied for BM25
    qid = row['query_id']
    qid_passages = validation_queries[validation_queries['query_id'] == qid] #find all the related pid's, passages etc, related to the current qid
    #Now we have all the passages that have the same qid lets iterate through them to calculate the cosine sim/BM25 scores before ranking them.
    for i, qrow in qid_passages.iterrows(): 
        query = qrow['query']
        passage = qrow['passage']
        pid = qrow['passage_id']
        relevance = qrow['relevance']
        # score, BM25score = cosine_similarity_BM25(query,passage,pid) #Combined by cosine similarity and BM25 functions so we only have to iterate through the passages once, saving time.
        BM25score = cosine_similarity_BM25(query,passage,pid) #Combined by cosine similarity and BM25 functions so we only have to iterate through the passages once, saving time.
        # score_stack = np.array([[qid,pid,score]]) #Create the qid,pid,score, row for the pid in question
        # curr_qid_pid_scores = np.vstack((curr_qid_pid_scores,score_stack)) #add it to the others for this qid
        
        BM25score_stack = np.array([[qid, pid, relevance, BM25score]]) #Repeat for BM25 with relevance
        curr_qid_pid_BM25scores = np.vstack((curr_qid_pid_BM25scores,BM25score_stack)) 

    # curr_qid_pid_scores = curr_qid_pid_scores[1:,:] #remove the row of zeros at the top of the temporary array used to initiate the array
    # curr_qid_pid_score = curr_qid_pid_scores[np.argsort(-curr_qid_pid_scores[:,-1])] #sort the current qid-pid pairs in ascending order (using the - in front) by their cosine similarity score
    # curr_qid_pid_score = curr_qid_pid_score[:100,:] #Take the top 100 pairs for this qid as requested
    # qid_pid_scores = np.vstack((qid_pid_scores,curr_qid_pid_score)) #Add them to the overall array for all qid's
    
    curr_qid_pid_BM25scores = curr_qid_pid_BM25scores[1:,:].astype(float)  #repeat the above for BM25
    curr_qid_pid_BM25score = curr_qid_pid_BM25scores[np.argsort(-curr_qid_pid_BM25scores[:,-1])] #-2 because now relevance is on the final column not BM25 score
    curr_qid_pid_BM25score = curr_qid_pid_BM25score[:100,:] #Top 100 scores (based on CW1 requirement to only take the top 100 for each qid)
    qid_pid_BM25scores = np.vstack((qid_pid_BM25scores,curr_qid_pid_BM25score))

# qid_pid_scores = qid_pid_scores[1:,:] #Remove the zeros at the top of the array that we used to initialise it
# # print(qid_pid_scores)
# np.savetxt("tfidf.csv", qid_pid_scores, fmt='%d,%d,%s', delimiter=',')

qid_pid_BM25scores = qid_pid_BM25scores[1:,:] 
np.savetxt("bm25.csv", qid_pid_BM25scores, fmt='%d,%d,%s,%s', delimiter=',')
np.save('qid_pid_BM25scores.npy', qid_pid_BM25scores)
# ##UNCOMMENT ONCE TASK FINISHED

qid_pid_BM25scores = np.load('qid_pid_BM25scores.npy', allow_pickle=True)
# ##UNCOMMENT ONCE TASK FINISHED

def mean_avg_precision(array):
    """Computes the mean average precision of an array of qids and relevance scores given the correct input format.
    Input: array (nxd) an array where the first column is query_id's, the last column is the ranking score for each query_id and the second to last column is a boolean as to whether the retrived document/passage is relevant or not.
    Output: (float) The average precision for the array of ranked queries.
    
    """
    #The function was first build for the BM25 array so I've kept in the specific lines for that array above each of the generic lines for the function.
    # unique_qids = np.unique(qid_pid_BM25scores[:,0]) #Get the unique qids so we can compute the average precision regardless of the length of the qid list
    unique_qids = np.unique(array[:,0]) #Get the unique qids so we can compute the average precision regardless of the length of the qid list
    
    count_relevant_qids = 0 #will be used later to calc the average precision
    query_average_precision = 0.0
    mean_average_precision = 0.0

    for qid in unique_qids: #iterate through each unique qid
        # current_qid = (qid_pid_BM25scores[:,0] == qid) #create an array of just the current relevant qid
        # bm25_cur_qid = qid_pid_BM25scores[current_qid]
        current_qid = (array[:,0] == qid) #create an array of just the current relevant qid
        bm25_cur_qid = array[current_qid]

        relevant_qids_mask = (bm25_cur_qid[:,-2] != 0) #create a mask based off of if the doc is relevant

        if np.sum(relevant_qids_mask) > 0: #If there are any relevant docs for this query_id 
            le = len(bm25_cur_qid + 1)
            arange = np.arange(1, len(bm25_cur_qid + 1)+1)
            cumsum = np.cumsum(relevant_qids_mask)
            running_precision = np.cumsum(relevant_qids_mask)/ np.arange(1, len(bm25_cur_qid + 1)+1)
            query_average_precision += np.sum(running_precision * relevant_qids_mask) / np.sum(relevant_qids_mask)
            count_relevant_qids += 1

    mean_average_precision += (query_average_precision / count_relevant_qids)

    return mean_average_precision 

BM25_MeanAP = mean_avg_precision(qid_pid_BM25scores)
print("BM25 Mean Average Precision: ", BM25_MeanAP)


def compute_ndcg(predictions, k=10):
    # sort the array by predicted ranking score
    sorted_idx = np.argsort(predictions[:, -1])[::-1]
    sorted_predictions = predictions[sorted_idx]

    ndcg_scores = []
    # iterate over queries
    for query_id in np.unique(sorted_predictions[:, 0]):
        query_mask = sorted_predictions[:, 0] == query_id
        query_relevance = sorted_predictions[query_mask][:, -2]

        #DCG
        gain = 2**query_relevance - 1
        discounts = np.log2(np.arange(len(query_relevance)) + 2)
        dcg = np.sum(gain[:k] / discounts[:k])

        #IDCG
        ideal_order = np.argsort(sorted_predictions[query_mask][:, -2])[::-1]
        ideal_gain = 2**query_relevance[ideal_order] - 1
        ideal_dcg = np.sum(ideal_gain[:k] / discounts[:k])

        # NDCG = DCG/IDCG, unless IDCG = 0 in which case it should be 0 so it doesnt give a nan value
        if ideal_dcg == 0:
            ndcg_scores.append(0)
        else:
            ndcg_scores.append(dcg / ideal_dcg)

    # average NDCG scores across queries
    return np.mean(ndcg_scores)

kay = 2
ndcg_BM25 = compute_ndcg(qid_pid_BM25scores, kay)
print("BM25 average NDCG: " ,ndcg_BM25)





# %%
