# coding: utf-8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import pearsonr
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string


def preprocess_text(text):
    """Preprocess one sentence: tokenizes, lowercases, applies the Porter stemmer,
     removes punctuation tokens and stopwords.
     Returns a string of tokens joined by whitespace."""
    stemmer = PorterStemmer()
    toks = word_tokenize(text)
    toks_stemmed = [stemmer.stem(tok.lower()) for tok in toks]
    toks_nopunc = [tok for tok in toks_stemmed if tok not in string.punctuation]
    toks_nostop = [tok for tok in toks_nopunc if tok not in set(stopwords.words('english'))]
    return " ".join(toks_nostop)


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


sts_dir = "../sts_strings/stsbenchmark/"
sts_train = f"{sts_dir}/sts-train.csv"
sts_dev = f"{sts_dir}/sts-dev.csv"

# load the texts
train_texts, train_y = load_sts(sts_train)
dev_texts, dev_y = load_sts(sts_dev)


# flatten the train texts to get a single list
# you have to memorize this confusing syntax ... or just iterate
flat_train_texts =[sent for pair in train_texts for sent in pair]

# create a TfidfVectorizer
# fit to each sentence from the training data as a separate document
vectorizer = TfidfVectorizer("content", lowercase=True, analyzer="word", use_idf=True, min_df=10)
vectorizer.fit(flat_train_texts)

print("Checking the vocabulary")
term_vocab = vectorizer.get_feature_names()
print(term_vocab[200:230])

print("Exploring sentence representations created by the vectorizer")
pair_reprs = vectorizer.transform(train_texts[0])
# a sparse datatype - saves only which positions are nonzero (where words are observed)
print(type(pair_reprs))
print(pair_reprs)
# compare the two representations
pair_similarity = cosine_similarity(pair_reprs[0], pair_reprs[1])
# similarity is returned in a matrix - have to get the right index to get a scalar
print(pair_similarity.shape)
print(pair_similarity[0,0])


# get cosine similarities of every pair in dev
cos_sims = []
for t1,t2 in dev_texts:
    pair_reprs = vectorizer.transform([t1,t2])
    pair_similarity = cosine_similarity(pair_reprs[0], pair_reprs[1])
    cos_sims.append(pair_similarity[0,0])


pearson = pearsonr(cos_sims, dev_y)
print(f"default settings: r={pearson[0]:.03}")


# Now try some preprocessing on texts
# Can normalization like removing stopwords remove differences that aren't meaningful?
preproc_train_texts = [preprocess_text(text) for text in flat_train_texts]

preproc_vectorizer = TfidfVectorizer("content", lowercase=True, analyzer="word",
                                     token_pattern="\S+", use_idf=True, min_df=10)
preproc_vectorizer.fit(preproc_train_texts)

# get cosine similarities of every pair in dev
cos_sims_preproc = []
for t1,t2 in dev_texts:
    t1_preproc = preprocess_text(t1)
    t2_preproc = preprocess_text(t2)
    pair_reprs = preproc_vectorizer.transform([t1_preproc, t2_preproc])
    pair_similarity = cosine_similarity(pair_reprs[0], pair_reprs[1])
    cos_sims_preproc.append(pair_similarity[0,0])


preproc_pearson = pearsonr(cos_sims_preproc, dev_y)
print(f"preprocessed text: r={preproc_pearson[0]:.03}")
