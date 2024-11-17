
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.preprocessing import LabelEncoder

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Получает часть речи для лемматизации."""
    if word.startswith('J'):
        return wordnet.ADJ
    elif word.startswith('V'):
        return wordnet.VERB
    elif word.startswith('N'):
        return wordnet.NOUN
    elif word.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    if pd.isna(text):
        return ' '
    text = text.lower()
    words = text.split()
    stop_words = set(stopwords.words('english'))

    filtered_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words if word not in stop_words]
    return ''.join(filtered_words)

def contains_url(text):
    if pd.isna(text):
        return 0
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return 1 if re.search(url_pattern, text) else 0

def contains_mobile(text):
    if pd.isna(text):
        return 0
    mobile_pattern = r'\+?\d[\d -]{7,}\d'
    return 1 if re.search(mobile_pattern, text) else 0

def get_label_encoder(y_train):
    le = LabelEncoder()
    le.fit(y_train)
    return le

def get_tfidf():
    trainDataFrame = pd.read_csv("datasets/train.csv")
    testDataFrame = pd.read_csv("datasets/test.csv")

    trainDataFrame = trainDataFrame[trainDataFrame["label"].isin(["ham", "spam"])]
    testDataFrame = testDataFrame[testDataFrame["label"].isin(["ham", "spam"])]

    trainDataFrame["email"] = trainDataFrame["email"].apply(clean_text)
    testDataFrame["email"] = testDataFrame["email"].apply(clean_text)

    trainDataFrame.dropna(subset=["email", "label"], inplace=True)
    testDataFrame.dropna(subset=["email", "label"], inplace=True)

    vectorizer = TfidfVectorizer(min_df=1, use_idf=True, ngram_range=(1, 1))

    X_train_matr = vectorizer.fit_transform(trainDataFrame["email"])
    y_train = trainDataFrame["label"]
    X_train = pd.DataFrame(X_train_matr.todense(), columns=vectorizer.get_feature_names_out())

    X_test_matr = vectorizer.transform(testDataFrame["email"])
    y_test = testDataFrame["label"]
    X_test = pd.DataFrame(X_test_matr.todense(), columns=vectorizer.get_feature_names_out())

    le = get_label_encoder(y_train)
    y_train_encoded = le.transform(y_train)
    y_test_encoded = le.transform(y_test)

    return X_train, X_test, y_train_encoded, y_test_encoded, vectorizer, le


