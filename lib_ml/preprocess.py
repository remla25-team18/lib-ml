# General imports
import re
import pandas as pd

# Natural Language imports
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download stopwords from nltk
nltk.download("stopwords")
nltk.download('wordnet')

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
    
    lemmatizer = WordNetLemmatizer()
    all_stopwords = stopwords.words("english")
    if "not" in all_stopwords:
        all_stopwords.remove("not")

    corpus = []
    for review in reviews:
        review = re.sub("[^a-zA-Z]", " ", review)
        review = review.lower().split()
        review = [lemmatizer.lemmatize(word) for word in review if word not in set(all_stopwords)]
        cleaned_review = " ".join(review)
        corpus.append(cleaned_review)
    
    return corpus

if __name__ == "__main__":
    # Example usage
    sample_reviews = [
        "This is a great restaurant!",
        "I did not like the food.",
        "Don't do it!!!!",
        "The service was excellent, but the food was terrible."
    ]
    processed_reviews = preprocess_text(sample_reviews)
    print(processed_reviews)