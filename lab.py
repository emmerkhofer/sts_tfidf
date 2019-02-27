# coding: utf-8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import pearsonr
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string


def load_sts(sts_data):
    # read the dataset
    texts = []
    labels = []

    with open(sts_data, 'r') as dd:
        for line in dd:
            fields = line.strip().split("\t")
            labels.append(float(fields[4]))
            t1 = fields[5].lower()
            t2 = fields[6].lower()
            texts.append((t1,t2))

    return texts, np.asarray(labels)


def preprocess_text(text):
    """Preprocess one sentence: tokenizes, lowercases, applies the Porter stemmer,
     removes punctuation tokens and stopwords.
     Returns a string of tokens joined by whitespace."""
    # work this function AFTER finishing the first part of the lab, without preprocessing
    # hint: look at the imports
    return text


sts_dir = "../sts_strings/stsbenchmark/"
sts_train = f"{sts_dir}/sts-train.csv"
sts_dev = f"{sts_dir}/sts-dev.csv"

# load the texts
train_texts, train_y = load_sts(sts_train)
dev_texts, dev_y = load_sts(sts_dev)

# create a TfidfVectorizer
# fit to each sentence from the training data as a separate document
# see the TfidfVectorizer docs for example input to the fit method
vectorizer = TfidfVectorizer("content", lowercase=True, analyzer="word", use_idf=True, min_df=10)


# get cosine similarities of every pair in dev
cos_sims = []
for t1,t2 in dev_texts:
    # use the vectorizer to get tfidf representations
    # use sklearn to get cosine similarity of the two representations
    cos_sims.append(0)

# raises a warning while all cos_sims are 0 - correlation is undefined when all values are same
pearson = pearsonr(cos_sims, dev_y)
print(f"default settings: r={pearson[0]:.03}")


# Now try some preprocessing on texts
# Can normalization like removing stopwords remove differences that aren't meaningful?

# fit to each sentence from the training data as a separate document, after preprocess_text
preproc_vectorizer = TfidfVectorizer("content", lowercase=True, analyzer="word",
                                     token_pattern="\S+", use_idf=True, min_df=10)

cos_sims_preproc = []
for t1,t2 in dev_texts:
    # use the vectorizer to get tfidf representations
    # use sklearn to get cosine similarity of the two representations
    cos_sims_preproc.append(0)


preproc_pearson = pearsonr(cos_sims_preproc, dev_y)
print(f"preprocessed text: r={preproc_pearson[0]:.03}")
