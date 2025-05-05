# General imports
import re
import pandas as pd

# Natural Language imports
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords from nltk
nltk.download("stopwords")

def preprocess_text(reviews):
    """
    Preprocesses a list of review texts.
    
    Parameters:
        - reviews: (List[str] or pd.DataFrame), raw reviews or a DataFrame with a 'Review' column.

    Returns:
        - List of cleaned, stemmed review texts.
    """
    if isinstance(reviews, pd.DataFrame):
        reviews = reviews["Review"].tolist()
    
    ps = PorterStemmer()
    all_stopwords = stopwords.words("english")
    if "not" in all_stopwords:
        all_stopwords.remove("not")

    corpus = []
    for review in reviews:
        review = re.sub("[^a-zA-Z]", " ", review)
        review = review.lower().split()
        review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
        corpus.append(" ".join(review))
    
    return corpus