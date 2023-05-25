import requests
import unicodedata
import re

def clean_name(name, stopwords=[]):
    name = unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode("utf-8")
    name=name.upper()
    name=name.replace("'S",'S')
    name=name.replace('-',' ')
    name=name.replace("'",' ')
    name=re.sub('[^A-Za-z0-9ñ\s]+', '', name) #remove special characters
    name=re.sub('\s{2}', ' ', name) #replace 2 white spaces to 1
    name=re.sub('\s{3}', ' ', name) #replace 2 white spaces to 1
    words=name.split()
    words=[w for w in words if w not in stopwords]
    name=" ".join(words)
    name=name.strip() #remove white spaces
    return str(name)


class Cleaner():
    def __init__(self) -> None:
        self.API_URL = "https://wkeryjfj2doknm3o.us-east-1.aws.endpoints.huggingface.cloud"
        self.headers = {"Authorization": "Bearer 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"}
        self.stopwords=['la','de','el','del','las','los']

    def query(self, payload):
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        return response.json()

    def clean(self, restaurant_name):
        cleaned_name = clean_name(restaurant_name, stopwords=self.stopwords)
        output = self.query(payload = {"inputs": f"REGULAR NAME: {cleaned_name}. CLEANED NAME:"})
        return output 