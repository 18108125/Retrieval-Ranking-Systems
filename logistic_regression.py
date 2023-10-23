import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import pandas as pd
import fasttext
import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')  # English
from tqdm import tqdm


model = fasttext.load_model('cc.en.300.bin')


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


def get_avg_embedding(sentence):

    words = preprocessing(sentence)
   
    embeddings = np.array([model[word] for word in words if word in model]) # Calculate embeddings for each word
    
    if embeddings.shape[0] > 0:
        return np.mean(embeddings, axis=0) #Average the embeddings for the whole sentence
    else:
        return np.zeros(model.get_dimension())


## DATA SOURCES FOR THIS TASK
#Create a dataframe for the tsv file - using pandas as it uses vectorised operations, e.g. .apply / .map - in the Q&A it was stated we'd need to use vectorised operations and parrallelisation
# all_passages = pd.read_csv('candidate_passages_top1000.tsv', sep='\t', header=None, names=['query_id', 'passage_id', 'query', 'passage'])
all_passages = pd.read_csv('candidate_passages_top1000.tsv', sep='\t', header=None, names=['query_id', 'passage_id', 'query', 'passage'])


#Read the test_queries file in the same way.
test_queries = pd.read_csv("test-queries.tsv", sep='\t', header=None, names=["query_id","query"])
validation_queries = pd.read_csv("validation_data.tsv", sep='\t', header=0, names=['query_id', 'passage_id', 'query', 'passage', 'relevance'])
train_queries = pd.read_csv("train_data.tsv", sep='\t', header=0, names=['query_id', 'passage_id', 'query', 'passage', 'relevance'])

# train_data = np.genfromtxt('train_data.tsv', delimiter='\t', dtype=str)
train_queries_np = train_queries.to_numpy()
validation_queries_np = validation_queries.to_numpy()

test_queries_np = test_queries.to_numpy()
test_queries_np_unique = np.unique(test_queries_np[:,0])

#%%
#Negative sampling

def gen_negative_samples(data, k):
    """ Generates a subset of negative samples given a numpy array with columns: qid, pid, query, passage, relevance"""
    unique_qids = np.unique(data[:, 0])
    samples = []
    for qid in unique_qids:
        pos_temp = data[(data[:, 0] == qid) & (data[:, -1] == 1)]
        neg_temp = data[(data[:, 0] == qid) & (data[:, -1] == 0)]
        samples.append(pos_temp[np.random.choice(pos_temp.shape[0], size=1, replace=False), :])
        if neg_temp.shape[0] < k:
            samples.append(neg_temp)
        else:
            samples.append(neg_temp[np.random.choice(neg_temp.shape[0], size=k, replace=False), :])
    new_data = np.concatenate(samples)
    return new_data


# neg_samples = gen_negative_samples(train_queries_np, 10)
# np.save('neg_samples.npy', neg_samples)
neg_samples = np.load('neg_samples.npy', allow_pickle=True)


queries = neg_samples[:, 2].astype(str)
passages = neg_samples[:, 3].astype(str)
relevance_scores = neg_samples[:, 4].astype(float)


queries_validation = validation_queries_np[:, 2].astype(str)
passages_validation = validation_queries_np[:, 3].astype(str)
relevance_scores_validation = validation_queries_np[:, 4].astype(float)


#UNCOMMENT ONCE DONE
# # Calculate embeddings for queries and passages
# query_embeddings_ns = np.array([get_avg_embedding(query) for query in tqdm(queries)])
# passage_embeddings_ns = np.array([get_avg_embedding(passage) for passage in tqdm(passages)])

# np.save('query_embeddings_ns.npy', query_embeddings_ns)
query_embeddings_ns = np.load('query_embeddings_ns.npy', allow_pickle=True)
# np.save('passage_embeddings_ns.npy', passage_embeddings_ns)
passage_embeddings_ns = np.load('passage_embeddings_ns.npy', allow_pickle=True)

# # Calculate embeddings for validation queries and passages
# query_embeddings_validation = np.array([get_avg_embedding(query) for query in tqdm(queries_validation)])
# np.save('query_embeddings_validation.npy', query_embeddings_validation)
# passage_embeddings_validation = np.array([get_avg_embedding(passage) for passage in tqdm(passages_validation)])
# np.save('passage_embeddings_validation.npy', passage_embeddings_validation)

query_embeddings_validation = np.load('query_embeddings_validation.npy', allow_pickle=True)
passage_embeddings_validation = np.load('passage_embeddings_validation.npy', allow_pickle=True)
#UNCOMMENT ONCE DONE



#OLD - FOR WHEN I WAS GOING TO USE ALL THE TRAINING DATA BUT IT TOOK TOO LONG
# np.save('query_embeddings.npy', query_embeddings)
# query_embeddings = np.load('query_embeddings.npy', allow_pickle=True)
# np.save('passage_embeddings.npy', passage_embeddings)
# passage_embeddings = np.load('passage_embeddings.npy', allow_pickle=True)

unique_passages = all_passages.drop_duplicates(['passage_id'])
unique_validation = validation_queries.drop_duplicates(['passage_id'])
unique_validation_qid = validation_queries.drop_duplicates(['query_id'])


#%% COSINE SIMILARITY


def cosine_similarity(query_embedding, passage_embedding):
    query_norms = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    passage_norms = np.linalg.norm(passage_embedding, axis=1, keepdims=True)
    # dot = np.dot(query_norms, passage_norms.T)
    dot = np.einsum('ij,ij->i', query_norms, passage_norms) #Einsum is more efficient than using np.dot which crashes my VS code
    numerator = np.einsum('ij,ij->i', query_embedding, passage_embedding)
    # numerator = np.dot(query_embedding, passage_embedding.T)
    similarity_scores = numerator / (dot + 1e-8)
    return similarity_scores

#UNCOMMENT ONCE DONE
# cosine_similarity_array = cosine_similarity(query_embeddings_ns, passage_embeddings_ns)
# np.save('cosine_similarity_array.npy', cosine_similarity_array)
# cosine_similarity_array_validation = cosine_similarity(query_embeddings_validation, passage_embeddings_validation)
# np.save('cosine_similarity_array_validation.npy', cosine_similarity_array_validation)
#UNCOMMENT ONCE DONE

cosine_similarity_array = np.load('cosine_similarity_array.npy', allow_pickle=True)
cosine_similarity_array_validation = np.load('cosine_similarity_array_validation.npy', allow_pickle=True)
# print(cosine_similarity_array)

query_length = np.char.count(queries[:], ' ')+1
passage_length = np.char.count(passages[:], ' ')+1

query_length_validation = np.char.count(queries_validation[:], ' ')+1
passage_length_validation = np.char.count(passages_validation[:], ' ')+1


#%% CREATE INPUTS FOR LOGISTIC REGRESSION

# xTr = np.concatenate(query_embeddings_ns,passage_embeddings_ns,cosine_similarity_array,query_length,passage_length)
xTr = np.stack((cosine_similarity_array,query_length,passage_length),axis=1)
yTr = relevance_scores

xTe = np.stack((cosine_similarity_array_validation,query_length_validation,passage_length_validation),axis=1)
yTe = relevance_scores_validation



#%% LOGISTIC REGRESSION

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # self.weights = np.zeros((X.shape[1], 1))
        self.weights = np.ones(X.shape[1])
        self.bias = 0

        # Gradient descent to min the CE loss function
        for i in range(self.num_iterations):
            # Compute the logits and probs of positive class
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)

            dw = (1 / X.shape[0]) * np.dot(X.T, (y_pred - y)) #np.einsum('ij,ij->i', query_norms, passage_norms)
            db = (1 / X.shape[0]) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw # update weights
            self.bias -= self.learning_rate * db #update bias

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(z)

        # Round the probabilities to the nearest integer (0 or 1) to make binary predictions
        class_predictions = np.round(y_pred).astype(int)
        return class_predictions, y_pred

    def _sigmoid(self, z):
        # Sigmoid function to map the logits to probabilities between 0 and 1
        return 1 / (1 + np.exp(-z))

#%% DO THE LOGISTIC REGRESSION

logistic_regression = LogisticRegression(learning_rate=0.01, num_iterations=1000)

logistic_regression.fit(xTr,yTr)

y_pred_train , probs_train = logistic_regression.predict(xTr)
y_pred_train, probs_train = np.expand_dims(y_pred_train, axis=1), np.expand_dims(probs_train, axis=1)
np.save('y_pred_train.npy', y_pred_train)
np.save('probs_train.npy', probs_train)

y_pred_validation, probs_validation = logistic_regression.predict(xTe)
accuracy = np.mean((y_pred_validation  == yTe.astype(int)))
print(f"Accuracy of model on validation data: {accuracy*100:.2f}%")

y_pred_validation, probs_validation = np.expand_dims(y_pred_validation, axis=1), np.expand_dims(probs_validation, axis=1)
np.save('y_pred_validation.npy', y_pred_validation)
np.save('probs_validation.npy', probs_validation)



LR_logits_validation_queries_np = np.hstack((validation_queries_np, probs_validation))

# #Sort array by logits
# sorted_indices = np.argsort(LR_logits_validation_queries_np[:,-1])[::-1]
# sorted_LR_logits_validation_queries_np = LR_logits_validation_queries_np[sorted_indices]
# unique_qids = np.unique(sorted_LR_logits_validation_queries_np[:, 0])

def rank_array_by_predicted_relevance(array):
    """Sort an array to get the top 100 predicted relevant passages for each qid. As long as the relevance score is in the final column and the query_id is in the first column"""

    #Sort array by logits (or whatever relevance prediction used)
    sorted_indices = np.argsort(array[:,-1])[::-1]
    sorted_array = array[sorted_indices]
    cols = array.shape[1]
    unique_qids = np.unique(array[:, 0])
    final_ranked_array = np.zeros((1,cols))

    for qid in unique_qids:                                         #NOTE TO SELF
        qid_stack = sorted_array[sorted_array[:,0] == qid][:100] #Do I need to add a thing where if there's less than 100 then to just add however many there are??
        final_ranked_array = np.concatenate((final_ranked_array,qid_stack))
    final_ranked_array = final_ranked_array[1:,:]
    return final_ranked_array

sorted_LR_logits_validation_queries_np = rank_array_by_predicted_relevance(LR_logits_validation_queries_np)
np.save('sorted_LR_logits_validation_queries_np.npy', sorted_LR_logits_validation_queries_np)



#%% EVALUATION METRICS

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
        selected_qid = array[current_qid]

        relevant_qids_mask = (selected_qid[:,-2] != 0) #create a mask based off of if the doc is relevant

        if np.sum(relevant_qids_mask) > 0: #If there are any relevant docs for this query_id 
            le = len(selected_qid)+1
            arange = np.arange(1, len(selected_qid) +1)
            cumsum = np.cumsum(relevant_qids_mask)
            running_precision = np.cumsum(relevant_qids_mask)/ np.arange(1, len(selected_qid) +1)
            query_average_precision += np.sum(running_precision * relevant_qids_mask) / np.sum(relevant_qids_mask)
            count_relevant_qids += 1

    mean_average_precision += (query_average_precision / count_relevant_qids)

    return mean_average_precision 


# def ndcg_at_k(scores, k):
#     """Compute normalized discounted cumulative gain (NDCG) at k."""
#     sorted_scores = np.sort(scores)[::-1]  # Sort scores in descending order
#     ideal_scores = np.sort(scores, kind='heapsort')[::-1][:k] #Added [:k] later  # Sort scores in descending order
#     dcg = np.sum(sorted_scores[:k] / np.log2(np.arange(2, k + 2)))  # Compute DCG
#     idcg = np.sum(ideal_scores / np.log2(np.arange(2, k + 2)))  # Compute ideal DCG
#     # idcg = np.sum(ideal_scores[:k] / np.log2(np.arange(2, k + 2)))  # Compute ideal DCG
#     return dcg / idcg if idcg > 0 else 0  # Return NDCG


# def compute_ndcg(ranked_array, k):
#     """Compute normalized discounted cumulative gain (NDCG) for a ranked array."""
#     ndcgs = []
#     for query_id in np.unique(ranked_array[:, 0]):
#         query_scores = ranked_array[ranked_array[:, 0] == query_id][:, -1] #changed from -2 to -1  # Extract relevance scores for the query
#         ndcgs.append(ndcg_at_k(query_scores, k))
#     return np.mean(ndcgs)  # Return average NDCG across all queries


def compute_ndcg(predictions, k=10):
    # sort the array by predicted ranking score
    sorted_idx = np.argsort(predictions[:, -1])[::-1]
    sorted_predictions = predictions[sorted_idx]

    ndcg_scores = []
    # iterate over queries
    for query_id in np.unique(sorted_predictions[:, 0]):
        query_mask = sorted_predictions[:, 0] == query_id
        query_relevance = sorted_predictions[query_mask][:, -2]

        # compute DCG
        gain = 2**query_relevance - 1
        discounts = np.log2(np.arange(len(query_relevance)) + 2)
        dcg = np.sum(gain[:k] / discounts[:k])

        # compute IDCG
        ideal_order = np.argsort(sorted_predictions[query_mask][:, -2])[::-1]
        ideal_gain = 2**query_relevance[ideal_order] - 1
        ideal_dcg = np.sum(ideal_gain[:k] / discounts[:k])

        # compute NDCG
        if ideal_dcg == 0:
            ndcg_scores.append(0)
        else:
            ndcg_scores.append(dcg / ideal_dcg)

    # average NDCG scores across queries
    return np.mean(ndcg_scores)


kay = 2

#%% USING THE EVALUATION METRICS FOR THE VALIDATION DATA

LR_MeanAP = mean_avg_precision(sorted_LR_logits_validation_queries_np)
print("Logistic Regression Mean Average Precision: ", LR_MeanAP)
sorted_ndcg_LR = compute_ndcg(sorted_LR_logits_validation_queries_np, kay)
print("Logistic Regression average NDCG: ", sorted_ndcg_LR)




#%% RERANKING CANDIDATE PASSAGES / TEST QUERIES

test_queries_np = test_queries.to_numpy()
all_passages_np = all_passages.to_numpy()
qids_test = test_queries_np[:,0]
qids_allpass = all_passages_np[:,0]
queries_allpass = all_passages_np[:, 2].astype(str)
passages_allpass = all_passages_np[:, 3].astype(str)

#UNCOMMENT ONCE DONE
# query_embeddings_allpass = np.array([get_avg_embedding(query) for query in tqdm(queries_allpass)])
# np.save('query_embeddings_allpass.npy', query_embeddings_allpass)
# passage_embeddings_allpass = np.array([get_avg_embedding(passage) for passage in tqdm(passages_allpass)])
# np.save('passage_embeddings_allpass.npy', passage_embeddings_allpass)
# cosine_similarity_array_allpass = cosine_similarity(query_embeddings_allpass, passage_embeddings_allpass)
# np.save('cosine_similarity_array_allpass.npy', cosine_similarity_array_allpass)
#UNCOMMENT ONCE DONE

query_embeddings_allpass = np.load('query_embeddings_allpass.npy', allow_pickle=True)
passage_embeddings_allpass = np.load('passage_embeddings_allpass.npy', allow_pickle=True)
cosine_similarity_array_allpass = np.load('cosine_similarity_array_allpass.npy', allow_pickle=True)

query_length_allpass = np.char.count(queries_allpass[:], ' ')+1
passage_length_allpass = np.char.count(passages_allpass[:], ' ')+1
xTTest = np.stack((cosine_similarity_array_allpass,query_length_allpass,passage_length_allpass),axis=1)

y_pred_test , probs_test = logistic_regression.predict(xTTest)
y_pred_test, probs_test = np.expand_dims(y_pred_test, axis=1), np.expand_dims(probs_test, axis=1)

LR_logits_test_queries_np = np.hstack((all_passages_np, probs_test))

sorted_LR_logits_test_queries_np = rank_array_by_predicted_relevance(LR_logits_test_queries_np)
np.save('sorted_LR_logits_test_queries_np.npy', sorted_LR_logits_test_queries_np)


#%% CREATE THE TEXT FILE REQUIRED
f = open("LR.txt", "w")
for i in range(test_queries_np.shape[0]):
    rank = 1
    qid = test_queries_np[i,0]
    pids_list_for_qid = sorted_LR_logits_test_queries_np[sorted_LR_logits_test_queries_np[:,0] == qid]
    for j in range(pids_list_for_qid.shape[0]):
        qid = str(qid)
        pid = str(pids_list_for_qid[j,1])
        score = str(pids_list_for_qid[j,-1])
        f.write(qid + "," + "A2" + "," + pid + "," + str(rank) + "," + score + "," + "LR" + "\n")
        rank += 1
f.close()

