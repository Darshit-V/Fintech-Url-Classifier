import json
import re
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

nltk.download('punkt')
nltk.download('stopwords')

def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, 'html.parser')
    text_content = soup.get_text()
    cleaned_text = text_content.replace('\n', '')
    cleaned_text = re.sub(r'[^\w\s\d]+', ' ', cleaned_text.encode('ascii', 'ignore').decode())
    return cleaned_text.strip() 

def preprocess_text(text):
    text = text.lower()

    tokens = word_tokenize(text)
 
    tokens = [word for word in tokens if word not in punctuation]

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and not word.isdigit()]
    
    return ' '.join(tokens)

def clean_json_data(input_file_path, output_file_path):
    # Read JSON data from input file
    with open(input_file_path, 'r') as f:
        data = json.load(f)

    with open(output_file_path, 'w') as txt_file:
        for entry in data:
            cleaned_text = clean_html(entry['raw_html'])

            preprocessed_text = preprocess_text(cleaned_text)

            txt_file.write(preprocessed_text + '\n')

input_file_path = 'scraped_data19.json'
output_file_path = 'preprocessed_data19.txt'

clean_json_data(input_file_path, output_file_path)
