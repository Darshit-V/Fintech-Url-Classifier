import scrapy
import requests
from bs4 import BeautifulSoup
import json
import re
import nltk
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader
import joblib


from new_new_new_model import BERTClassifier


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


nltk.download('punkt')
nltk.download('stopwords')

class MyCrawler(scrapy.Spider):
    name = 'mycrawler'
    allowed_domains = ['www.britannica.com']
    start_urls = ['https://www.britannica.com/sports/sports']  

    def parse(self, response):
        self.logger.info('Parsing URL: %s', response.url)
        
        with open('links17.txt', 'a') as f:
            f.write(response.url + '\n')
        
        
        scraped_data = self.scrape_html(response.url)
        if scraped_data:
            
            with open('scraped_data7.json', 'a') as json_file:
                if json_file.tell() == 0:  
                    json_file.write('[')  
                else:
                    json_file.write(',')  
                json.dump(scraped_data, json_file, indent=4)
                json_file.write('\n')  

    def scrape_html(self, url):
        
        response = requests.get(url)

        
        if response.status_code == 200:
            
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
      
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', text)

     
        cleantext = cleantext.lower()

        return cleantext

    def predict_fintech(self, text):
       
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
      
        label_encoder = joblib.load('label_encoder.pkl')
        
      
        model = BERTClassifier(num_classes=len(label_encoder.classes_))  
        model.load_state_dict(torch.load('bert_model_updated3.pt'))
        
 
        model.eval()

      
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

     
        predicted_label = torch.argmax(outputs, dim=1).item()

        return predicted_label

    def close(self, reason):
    
        with open('scraped_data7.json', 'a') as json_file:
            json_file.write(']')


if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess


    process = CrawlerProcess()


    process.crawl(MyCrawler)

   
    process.start()
