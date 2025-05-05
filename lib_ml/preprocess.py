# General imports
import re
import pandas as pd

# Natural Language imports
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords from nltk
nltk.download("stopwords")


def preprocess(file : str):
    """Preprocesses the given dataset.

    Parameters:
        - file: (str), path to the dataset file

    Returns:
        - tuple (Preprocessed reviews, respective labels)    
    """
    dataset = pd.read_csv(file, delimiter="\t", quoting=3)

    ps = PorterStemmer()
    all_stopwords = stopwords.words("english")
    if "not" in all_stopwords:
        all_stopwords.remove("not")

    corpus = []
    for review in dataset["Review"]:
        review = re.sub("[^a-zA-Z]", " ", review)
        review = review.lower().split()
        review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
        corpus.append(" ".join(review))

    return corpus, dataset.iloc[:, -1].values