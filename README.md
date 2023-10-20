# Retrieval-Ranking-Systems
Information Retrieval and Data Mining Coursework

# Files & their purpose:

1) Text_statistics: Extracts terms (1-grams) from the raw text. It then implements a function that counts the number of occurences of terms in the data set, plots their probability of occurence (normalised frequency) against their frequency ranking, and assesses whether the terms follow Zipf's law.

2) Inverted_index: Uses the vocabulary of terms identified in Text_statistics to build an inverted index so that passages can be retrieved in an efficient way.

3) Retrieval_models: Extracts the TF-IDF vector representations of the passages using the inverted index. Then a basic vector space model with TF-IDF and cosine similarity is used to retrieve at most 100 passages from within the 1000 candidate passages for each query. The outcomes are stored in a file named tfidf.csv with the format qid,pid,score. The qid,pid pairs are ranked by similarity score in descending order. The inverted index is also used to implement BM25 to retrieve at most 100 passages.

4) Query_likelihood_language_models: Three query likelihood lanague models are implemented, one with Laplace smoothing, another with Lidstone correction, and one more with Dirichlet smoothing. The natural logarithm of the probability scores are used. This is interpreted as being the likelihood of a document being relevant given a query.

# The data set consists of 3 files:
test-queries.tsv - is a tab separated file, where each row contains a test query identifier and the actual query text.
passage-collection.txt - is a collection of passages, one per row.
candidate-passages-top1000.tsv is a tab separated file with an initial selection of at most 1000 passages for each of the queries in test-queries.tsv. 
