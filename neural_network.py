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
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

embed_model = fasttext.load_model('cc.en.300.bin')


def preprocessing(passage,stop_words=True): #Easiest/quickest way to toggle the removal of stopwords for the rest of the file is by using this line.
    """Function that takes sentences as input and preprocesses them to output tokens.
    There is the option to remove stopwords, which the default is set to True.

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
   
    embeddings = np.array([embed_model[word] for word in words if word in embed_model]) # Calculate embeddings for each word
    
    if embeddings.shape[0] > 0:
        return np.mean(embeddings, axis=0) #Average the embeddings for the whole sentence
    else:
        return np.zeros(embed_model.get_dimension())


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

print(np.count_nonzero(train_queries_np[:, -1] == 1))
print(len(train_queries_np[:,-1]))

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


neg_samples = gen_negative_samples(train_queries_np, 10)
np.save('neg_samples.npy', neg_samples)
neg_samples = np.load('neg_samples.npy', allow_pickle=True)
relevance_scores = neg_samples[:, 4].astype(float)
x_train_pre_resampling = neg_samples[:, 0:4]

over = RandomOverSampler(sampling_strategy=0.4, random_state=1)
under = RandomUnderSampler(sampling_strategy=0.5, random_state=1)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

x_train, y_train = pipeline.fit_resample(x_train_pre_resampling, relevance_scores)

# print(np.count_nonzero(y_train[:] == 1))

qids = neg_samples[:,0]
queries = neg_samples[:, 2].astype(str)
passages = neg_samples[:, 3].astype(str)
# print(np.count_nonzero(relevance_scores[:] == 1))
# print(len(neg_samples[:,-1]))

# relevance_scores = neg_samples[1:, 4].astype(float)

qids_validation = validation_queries_np[:,0]
queries_validation = validation_queries_np[:, 2].astype(str)
passages_validation = validation_queries_np[:, 3].astype(str)
relevance_scores_validation = validation_queries_np[:, 4].astype(float)


#UNCOMMENT ONCE DONE
# Calculate embeddings for queries and passages
query_embeddings_ns = np.array([get_avg_embedding(query) for query in tqdm(queries)])
passage_embeddings_ns = np.array([get_avg_embedding(passage) for passage in tqdm(passages)])
#UNCOMMENT ONCE DONE
np.save('query_embeddings_ns.npy', query_embeddings_ns)
query_embeddings_ns = np.load('query_embeddings_ns.npy', allow_pickle=True)
np.save('passage_embeddings_ns.npy', passage_embeddings_ns)
passage_embeddings_ns = np.load('passage_embeddings_ns.npy', allow_pickle=True)
#UNCOMMENT ONCE DONE
# Calculate embeddings for validation queries and passages
query_embeddings_validation = np.array([get_avg_embedding(query) for query in tqdm(queries_validation)])
passage_embeddings_validation = np.array([get_avg_embedding(passage) for passage in tqdm(passages_validation)])

np.save('query_embeddings_ns.npy', query_embeddings_validation)
query_embeddings_validation = np.load('query_embeddings_validation.npy', allow_pickle=True)
np.save('passage_embeddings_validation.npy', passage_embeddings_validation)
passage_embeddings_validation = np.load('passage_embeddings_validation.npy', allow_pickle=True)
#UNCOMMENT ONCE DONE


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
cosine_similarity_array = cosine_similarity(query_embeddings_ns, passage_embeddings_ns)
np.save('cosine_similarity_array.npy', cosine_similarity_array)
cosine_similarity_array_validation = cosine_similarity(query_embeddings_ns, passage_embeddings_ns)
np.save('cosine_similarity_array_validation.npy', cosine_similarity_array_validation)
#UNCOMMENT ONCE DONE

cosine_similarity_array = np.load('cosine_similarity_array.npy', allow_pickle=True)
cosine_similarity_array_validation = np.load('cosine_similarity_array_validation.npy', allow_pickle=True)
# print(cosine_similarity_array)

query_length = np.char.count(queries[:], ' ')+1
passage_length = np.char.count(passages[:], ' ')+1

query_length_validation = np.char.count(queries_validation[:], ' ')+1
passage_length_validation = np.char.count(passages_validation[:], ' ')+1

#%% CREATE INPUTS FOR XGBOOST/LAMBDAMART (reused for the Neural Network as requested)

# xTr = np.concatenate(query_embeddings_ns,passage_embeddings_ns,cosine_similarity_array,query_length,passage_length)
xTr_full = np.stack((cosine_similarity_array,query_length,passage_length, relevance_scores), axis=1)
xTr_full = np.stack((cosine_similarity_array,query_length,passage_length, relevance_scores), axis=1)

xTr_full, xTr_validation = train_test_split(xTr_full, test_size=0.1, stratify=xTr_full[:,-1])
# print(np.count_nonzero(xTr_full[:, -1] == 1))
xTr_full_pre_resample  = xTr_full[:,:-1]
y_train_pre_resampled = xTr_full[:,-1].astype(int)


over = RandomOverSampler(sampling_strategy=0.4, random_state=1)
under = RandomUnderSampler(sampling_strategy=0.5, random_state=1)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

x_train_resampled, y_train_resampled = pipeline.fit_resample(xTr_full_pre_resample, y_train_pre_resampled)
# print(np.count_nonzero(y_train_resampled[:] == 1))


xTr_validation, yTr_validation = xTr_validation[:,:-1], xTr_validation[:,-1]

xTe = np.stack((cosine_similarity_array_validation,query_length_validation,passage_length_validation),axis=1)
yTe = relevance_scores_validation

stdscaler = StandardScaler()

stdscaler.fit(x_train_resampled)

x_train_resampled_scaled = stdscaler.transform(x_train_resampled)
xTr_validation_scaled = stdscaler.transform(xTr_validation)
xTe_scaled = stdscaler.transform(xTe)




class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 20)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(20, 20)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(20, 20)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc4(x))
        return x

# create the model
input_dim = x_train_resampled_scaled.shape[1]
model = Net(input_dim)

# define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

batch_size_list =  [100, 150, 200, 250, 300]
num_epochs_list =  [4, 8, 10, 12]

best_accuracy = 0.0
best_epochs_track = 0.0
best_batch_size_track = 0.0

#FIND OUT THE BEST HYPER PARAMETER VALUES - Keep me in!!
# for epoch_iter in num_epochs_list:
#     for batch_iter in batch_size_list:
#         # training loop
#         for epoch in tqdm(range(epoch_iter)):
#             running_loss = 0.0
#             for i in range(0, len(x_train_resampled_scaled), batch_iter):
#                 # get the inputs
#                 inputs = torch.from_numpy(x_train_resampled_scaled[i:i+batch_iter]).type(torch.float32)
#                 labels = torch.from_numpy(y_train_resampled[i:i+batch_iter]).type(torch.float32)
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#                 # forward + backward + optimize
#                 outputs = model(inputs)
#                 loss = criterion(outputs.squeeze(1), labels)
#                 loss.backward()
#                 optimizer.step()
#                 # print statistics
#                 running_loss += loss.item()
#             print(f"Epoch {epoch+1}, Loss: {running_loss/len(x_train_resampled_scaled)}")

#         # evaluation
#         with torch.no_grad():
#             inputs = torch.from_numpy(xTr_validation_scaled).type(torch.float32)
#             labels = torch.from_numpy(yTr_validation).type(torch.float32)
#             outputs = model(inputs)
#             accuracy = torch.mean(((outputs > 0.5) == labels).type(torch.float32)).item()
#             print(f"Accuracy of model on validation data: {accuracy*100:.2f}%")

#             if accuracy > best_accuracy:
#                     best_accuracy = accuracy
#                     best_epochs_tracker = epoch_iter
#                     best_batch_size = batch_iter
                    

#         print(f"Best validation accuracy: {best_accuracy*100:.2f}% (epoch {best_epochs_tracker}, batch size {best_batch_size})")
batch_size = 150
# training loop
for epoch in tqdm(range(10)):
    running_loss = 0.0
    for i in range(0, len(x_train_resampled_scaled), batch_size):
        # get the inputs
        inputs = torch.from_numpy(x_train_resampled_scaled[i:i+batch_size]).type(torch.float32)
        labels = torch.from_numpy(y_train_resampled[i:i+batch_size]).type(torch.float32)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(1), labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(x_train_resampled_scaled)}")

# evaluation
with torch.no_grad():
    inputs = torch.from_numpy(xTr_validation_scaled).type(torch.float32)
    labels = torch.from_numpy(yTr_validation).type(torch.float32)
    outputs = model(inputs)
    accuracy = torch.mean(((outputs > 0.5) == labels).type(torch.float32)).item()
    print(f"Accuracy of model on validation data: {accuracy*100:.2f}%")

    if accuracy > best_accuracy:
            best_accuracy = accuracy
            # best_epochs_tracker = epoch_iter
            # best_batch_size = batch_iter
            

# print(f"Best validation accuracy: {best_accuracy*100:.2f}% (epoch {best_epochs_tracker}, batch size {best_batch_size})")

def predict(model, input_data):
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_data).type(torch.float32)
        output_probabilities = model(input_tensor)
    return output_probabilities.numpy()

# best_model_state_dict = torch.load("best_model.pt")

# create a new instance of the Net class and load the saved state dictionary
# best_model = Net(input_dim)
# best_model.load_state_dict(best_model_state_dict)

test_predictions = predict(model, xTe_scaled)

NN_validation_queries_np = np.hstack((validation_queries_np, test_predictions))


def rank_array_by_predicted_relevance(array):
    """Sort an array to get the top 100 predicted relevant passages for each qid. As long as the relevance score is in the final column and the query_id is in the first column"""

    #Sort array by logits (or whatever relevance prediction used)
    sorted_indices = np.argsort(array[:,-1])[::-1]
    sorted_array = array[sorted_indices]
    cols = array.shape[1]
    unique_qids = np.unique(array[:, 0])
    final_ranked_array = np.zeros((1,cols))

    for qid in unique_qids:                                         #NOTE TO SELF
        qid_stack = sorted_array[sorted_array[:,0] == qid][:100,:] #Do I need to add a thing where if there's less than 100 then to just add however many there are??
        final_ranked_array = np.concatenate((final_ranked_array,qid_stack))
    final_ranked_array = final_ranked_array[1:,:]
    return final_ranked_array


#UNCOMMENT ONCE DONE - MAY WANT TO UNCOMMENT THIS ONE SOONER
sorted_NN_validation_queries_np = rank_array_by_predicted_relevance(NN_validation_queries_np)
np.save('sorted_NN_validation_queries_np.npy', sorted_NN_validation_queries_np)
#UNCOMMENT ONCE DONE
sorted_NN_validation_queries_np = np.load('sorted_NN_validation_queries_np.npy', allow_pickle=True)


#%% EVALUATION METRICS

def mean_avg_precision(array):
    """Computes the mean average precision of an array of qids and relevance scores given the correct input format.
    Input: array (nxd) an array where the first column is query_id's, the last column is the ranking score for each query_id and the second to last column is a boolean as to whether the retrived document/passage is relevant or not.
    Output: (float) The average precision for the array of ranked queries.
    
    """
    #The function was first built for the BM25 array so I've kept in the specific lines for that array above each of the generic lines for the function.
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

NN_MeanAP = mean_avg_precision(sorted_NN_validation_queries_np)
print("Neural Network Mean Average Precision: ", NN_MeanAP)
ndcg_NN = compute_ndcg(sorted_NN_validation_queries_np, kay)
print("Neural Network average NDCG: ", ndcg_NN)


#%% RERANKING CANDIDATE PASSAGES/TEST QUERIES
##
test_queries_np = test_queries.to_numpy()
all_passages_np = all_passages.to_numpy()
qids_test = test_queries_np[:,0]
qids_allpass = all_passages_np[:,0]
queries_allpass = all_passages_np[:, 2].astype(str)
passages_allpass = all_passages_np[:, 3].astype(str)

#UNCOMMENT ONCE DONE
query_embeddings_allpass = np.array([get_avg_embedding(query) for query in tqdm(queries_allpass)])
np.save('query_embeddings_allpass.npy', query_embeddings_allpass)
passage_embeddings_allpass = np.array([get_avg_embedding(passage) for passage in tqdm(passages_allpass)])
np.save('passage_embeddings_allpass.npy', passage_embeddings_allpass)
cosine_similarity_array_allpass = cosine_similarity(query_embeddings_allpass, passage_embeddings_allpass)
np.save('cosine_similarity_array_allpass.npy', cosine_similarity_array_allpass)
#UNCOMMENT ONCE DONE

query_embeddings_allpass = np.load('query_embeddings_allpass.npy', allow_pickle=True)
passage_embeddings_allpass = np.load('passage_embeddings_allpass.npy', allow_pickle=True)
cosine_similarity_array_allpass = np.load('cosine_similarity_array_allpass.npy', allow_pickle=True)

query_length_allpass = np.char.count(queries_allpass[:], ' ')+1
passage_length_allpass = np.char.count(passages_allpass[:], ' ')+1
xTTest = np.stack((cosine_similarity_array_allpass,query_length_allpass,passage_length_allpass),axis=1)

xTTest_scaled = stdscaler.transform(xTTest)

Final_test_predictions = predict(model, xTTest_scaled)

NN_Test_queries_np = np.hstack((all_passages_np, Final_test_predictions))

sorted_NN_Test_queries_np = rank_array_by_predicted_relevance(NN_Test_queries_np)
np.save('sorted_NN_Test_queries_np.npy', sorted_NN_Test_queries_np)
sorted_NN_Test_queries_np = np.load('sorted_NN_Test_queries_np.npy', allow_pickle=True)


#%% CREATE THE TEXT FILE REQUIRED
f = open("NN.txt", "w")
for i in range(test_queries_np.shape[0]):
    rank = 1
    qid = test_queries_np[i,0]
    pids_list_for_qid = sorted_NN_Test_queries_np[sorted_NN_Test_queries_np[:,0] == qid]
    for j in range(pids_list_for_qid.shape[0]):
        qid =str(qid)
        pid = str(pids_list_for_qid[j,1])
        score = str(pids_list_for_qid[j,-1])
        f.write(qid + "," + "A2" + "," + pid + "," + str(rank) + "," + score + "," + "NN" + "\n")
        rank += 1
f.close()
