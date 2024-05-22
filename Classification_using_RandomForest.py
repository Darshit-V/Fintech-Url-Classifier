import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
import requests
from bs4 import BeautifulSoup
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib


from joblib import load

rf_classifier = load('RandomForest_model.joblib')


nltk.download('punkt')
nltk.download('stopwords')

class MyCrawler(CrawlSpider):
    name = 'mycrawler'
    start_urls = ['https://www.britannica.com/sports/sports']  

    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )

    custom_settings = {
        'DEPTH_LIMIT': 0,
        'SCHEDULER_DISK_QUEUE': 'scrapy.squeues.PickleFifoDiskQueue',
        'SCHEDULER_MEMORY_QUEUE': 'scrapy.squeues.FifoMemoryQueue'
    }

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse_item)    

    def parse_item(self, response):
        self.logger.info('Parsing URL: %s', response.url)
        with open('links15.txt', 'a') as f:
            f.write(response.url + '\n')
        
        
        scraped_data = self.scrape_html(response.url)
        if scraped_data:
            with open('Prediction_by_RandomForest.json', 'a') as json_file:
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

        tokens = word_tokenize(cleantext)

        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and not word.isdigit()]

        return ' '.join(tokens)

    def predict_fintech(self, text):
        
        rf_classifier = joblib.load('RandomForest_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer_RandomForest.joblib')
        X = vectorizer.transform([text])

        prediction = rf_classifier.predict(X)

        prediction = int(prediction[0])

        return prediction

    def close(self, reason):
        with open('Prediction_by_RandomForest.json', 'a') as json_file:
            json_file.write(']')

if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess

    process = CrawlerProcess()

    process.crawl(MyCrawler)

    process.start()
