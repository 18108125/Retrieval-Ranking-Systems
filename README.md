# Retrieval-Ranking-Systems - files & their purpose
Information Retrieval and Data Mining Coursework 1 & 2

# Basic Models (CW1):

1) Text_statistics: Extracts terms (1-grams) from the raw text. It then implements a function that counts the number of occurences of terms in the data set, plots their probability of occurence (normalised frequency) against their frequency ranking, and assesses whether the terms follow Zipf's law.

2) Inverted_index: Uses the vocabulary of terms identified in Text_statistics to build an inverted index so that passages can be retrieved in an efficient way.

3) Retrieval_models: Extracts the TF-IDF vector representations of the passages using the inverted index. Then a basic vector space model with TF-IDF and cosine similarity is used to retrieve at most 100 passages from within the 1000 candidate passages for each query. The outcomes are stored in a file named tfidf.csv with the format qid,pid,score. The qid,pid pairs are ranked by similarity score in descending order. The inverted index is also used to implement BM25 to retrieve at most 100 passages.

4) Query_likelihood_language_models: Three query likelihood lanague models are implemented, one with Laplace smoothing, another with Lidstone correction, and one more with Dirichlet smoothing. The natural logarithm of the probability scores are used. This is interpreted as being the likelihood of a document being relevant given a query.

# More Advanced Models (CW2):

1) Evaluating_retrieval_quality: Manually implements methods to compute the average precision and NDCG metrics to assess the performance of the BM25 retrieval model on the validation data (the coursework required the performance to be assessed on the validation data).

2) Logistic_regression: Represents the passages and queries based on FastText word embeddings, the word embeddings are then averaged across the query/passage to represent query/passage level embeddings. These are input into a manual implementation of Logistic Regression, the performance of the trained model is evaulated on the validation data using the metrics described above (the coursework required the performance to be assessed on the validation data). Negative sampling was implemented during training.

3) LambdaMART: Uses the LambdaMART learning to take algorithm from the XGBoost gradient boosting library to learn a model that can rerank passages. The performance of the trained model is evaulated on the validation data using the metrics described above (the coursework required the performance to be assessed on the validation data).

4) Neural_network: Implements a sequential feed-forward network using PyTorch, which has four fully connected (dense) layers. The hidden layers have 20 hidden units each and are activated by the widely used ReLU activation function after each fully connected layer. Each hidden layer is followed by a dropout layer, which randomly sets 20% of the input units to zero during training.

# The data set consists of 5 files:
test-queries.tsv - is a tab separated file, where each row contains a test query identifier and the actual query text.
passage-collection.txt - is a collection of passages, one per row.
candidate-passages-top1000.tsv is a tab separated file with an initial selection of at most 1000 passages for each of the queries in test-queries.tsv. 
train_data.tsv is used as a training set for the models, it includes a 'relevance' column indicating the relevance of the passage to the query, in addition to the qid, pid, query and passage columns. 
validation_data.tsv is used as a validation set for the models, it includes a 'relevance' column indicating the relevance of the passage to the query, in addition to the qid, pid, query and passage columns.
