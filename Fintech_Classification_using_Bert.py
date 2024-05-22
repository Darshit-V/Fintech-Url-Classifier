import scrapy
import requests
from bs4 import BeautifulSoup
import json
import re
import nltk
import torch
from transformers import BertTokenizer
import joblib
from new_new_new_model import BERTClassifier  # Import your BERTClassifier model here
import sys
print("file loaded")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
label_encoder = joblib.load('label_encoder.pkl')
model = BERTClassifier(num_classes=len(label_encoder.classes_))
model.load_state_dict(torch.load('bert_model_updated3.pt'))
model.eval()

nltk.download('punkt')
nltk.download('stopwords')


class MyCrawler(scrapy.Spider):
    print("startedcrawilng")
    name = 'mycrawler'
    # allowed_domains = ['www.britannica.com']
    start_urls = []  # Empty start_urls list

    def __init__(self, *args):
        super(MyCrawler, self).__init__(*args)
        if len(sys.argv) > 1:
            self.start_urls.append(sys.argv[1])  # Add URL from command-line argument

    def parse(self, response):
        print('Parsing URL:', response.url)

        with open('links17.txt', 'a') as f:
            f.write(response.url + '\n')

        scraped_data = self.scrape_html(response.url)
        if scraped_data:
            with open('scraped_datatest1.json', 'a') as json_file:
                if json_file.tell() == 0:
                    json_file.write('[')
                else:
                    json_file.write(',')
                json.dump(scraped_data, json_file, indent=4)
                json_file.write('\n')

    def scrape_html(self, url):
        print('Scraping HTML for URL:', url)
        response = requests.get(url)

        if response.status_code == 200:
            print('Successfully fetched URL:', url)
            soup = BeautifulSoup(response.content, 'html.parser')
            raw_html = str(soup)
            preprocessed_text = self.preprocess_text(raw_html)
            prediction = self.predict_fintech(preprocessed_text)

            data = {
                "url": url,
                "raw_html": raw_html,
                "prediction": prediction
            }
            return data
        else:
            print("Failed to fetch URL:", url)
            return None

    def preprocess_text(self, text):
        print('Preprocessing text:', text)
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', text)
        cleantext = cleantext.lower()
        return cleantext

    def predict_fintech(self, text):
        print('Predicting fintech:', text)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        model.eval()

        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

        predicted_label = torch.argmax(outputs, dim=1).item()
        print('Predicted label:', predicted_label)
        return predicted_label

    def close(self, reason):
        with open('scraped_datatest1.json', 'a') as json_file:
            json_file.write(']')


if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess

    # Check if URL argument is provided
    if len(sys.argv) < 2:
        print("Usage: python your_python_script.py <URL>")
        sys.exit(1)

    process = CrawlerProcess()
    process.crawl(MyCrawler)
    process.start()
