import json
import pandas as pd
import time
from tqdm import tqdm 
from google.cloud import language_v2


class SentimentAnalyis:
    def __init__(self) -> None:
        self.data = pd.read_csv("./SMSSpamCollection", sep="\t", header=None, 
                 names=["Label", "Message"])
    

    def load_sentiment_data(self):
        with open("sentiment_data.json", "r") as infile:
            return json.load(infile)

    def find_sentiment(self):
        client = language_v2.LanguageServiceClient()
        document_type_in_plain_text = language_v2.Document.Type.PLAIN_TEXT
        language_code = "en"
        self.sentiment_data = self.load_sentiment_data()
        work_already_done = list(self.sentiment_data.keys())
        for index, row in tqdm(self.data.iterrows(), total=len(self.data)):
            if index not in work_already_done:
                required_result = {}
                try:
                    document = {
                        "content": row['Message'],
                        "type_": document_type_in_plain_text,
                        "language_code": language_code,
                    }
                    response = client.analyze_sentiment(
                        request={"document": document}
                    )
                    required_result = {
                        'data' : row["Message"],
                        'sentiment_score' : response.document_sentiment.score,
                        'sentiment_magnitude': response.document_sentiment.magnitude,
                        'sentences' : {}
                    }
                    counter = 0
                    for sentence in response.sentences:
                        required_result["sentences"][counter] = {
                            "context" : sentence.text.content,
                            "score" : sentence.sentiment.score,
                            "magnitude" : sentence.sentiment.magnitude
                        }
                        counter += 1
                except Exception as e :
                    print(f"Error analyzing sentiment for index {index}: {e}")
                    break

                self.sentiment_data[index]= required_result
                time.sleep(0.05)
            else:
                print(f"Work for {index} is already done.")
        with open("sentiment_data.json", "w") as outfile: 
            json.dump(self.sentiment_data, outfile)

    def verify_sentiement(self):
        self.sentiment_data = self.load_sentiment_data()
        print(f"result list size: {len(self.sentiment_data)} vs data list size: {len(self.data)}")
    
    def get_sentiment_data(self):
        self.sentiment_data = self.load_sentiment_data()
        sentiment = []
        for key,value in self.sentiment_data.items():
            sentiment.append((value.get('sentiment_score'),value.get('sentiment_magnitude')))
        return sentiment
