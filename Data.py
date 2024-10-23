import pandas as pd
import nltk
import string
import numpy as np
import json
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self,location) -> None:
        self.location = location
        self.data = None

    def load_sentiment_data(self):
        with open("sentiment_data.json", "r") as infile:
            return json.load(infile)
        
    def get_sentiment_data(self):
        self.sentiment_data = self.load_sentiment_data()
        sentiment = []
        for key,value in self.sentiment_data.items():
            sentiment.append((value.get('sentiment_score'),value.get('sentiment_magnitude')))
        return sentiment
        
    def load_data(self):
        def remove_punctuation_and_tokenize(text):
            try:
                stop_words = nltk.corpus.stopwords.words('english')
            except LookupError:
                nltk.download('stopwords')
                nltk.download('punkt')
                stop_words = nltk.corpus.stopwords.words('english')
            text_without_punctuation = text.translate(str.maketrans('', '', string.punctuation))
            tokens = nltk.tokenize.word_tokenize(text_without_punctuation)
            filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
            return " ".join(filtered_tokens)
        self.data = pd.read_csv(self.location, sep="\t", header=None, 
                 names=["Label", "Message"])
        
        self.data['Message_after_preprocessing'] = self.data["Message"].apply(remove_punctuation_and_tokenize)

        sentiment_data = self.get_sentiment_data()
        self.data['Sentiment_Analysis'] = sentiment_data

    def oversample_data(self,X,Y):
        smote = SMOTE()
        x_resampled, y_resampled = smote.fit_resample(X, Y)
        return (x_resampled, y_resampled)

        
    def data_convertor_tfidf(self,sentiment):
        vectorizer = TfidfVectorizer()
        message_vector = vectorizer.fit_transform(self.data["Message_after_preprocessing"])
        if sentiment:
            sentiment_data = self.data["Sentiment_Analysis"].tolist()
            sentiment_data = np.array(sentiment_data).reshape(-1, 2)
            message_vector = np.hstack((message_vector.toarray(), sentiment_data)) 
        encoder = LabelEncoder()
        encoder.fit(["ham","spam"])
        required_labels = encoder.transform(self.data["Label"])
        return (message_vector,required_labels)

    def data_convertor(self,sentiment):
        vectorizer = CountVectorizer()
        message_vector = vectorizer.fit_transform(self.data["Message"])
        if sentiment:
            sentiment_data = self.data["Sentiment_Analysis"].tolist()
            sentiment_data = np.array(sentiment_data).reshape(-1, 2)
            message_vector = np.hstack((message_vector.toarray(), sentiment_data))
        encoder = LabelEncoder()
        encoder.fit(["ham","spam"])
        required_labels = encoder.transform(self.data["Label"])
        return (message_vector,required_labels)
        
    
    def data_split(self,x,y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        return ((x_train,y_train),(x_test,y_test))


    def get_data(self):
        return self.data
    

    def plot_and_save_confusion_matrix(self,cm, title, filename):
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap='Blues')

        # Add labels and title
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_xticks(range(len(cm[0])))
        ax.set_yticks(range(len(cm)))

        ax.set_xticklabels(['ham', 'spam'])
        ax.set_yticklabels(['ham', 'spam'])
    

        # Add text with values (optional)
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=8)

        fig.colorbar(im)
        fig.tight_layout()

        # Save the plot as an image
        plt.savefig(f"./output/{filename}")
        plt.close()


