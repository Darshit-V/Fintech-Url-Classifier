import json
import re
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

# Download NLTK resources (run only once)
nltk.download('punkt')
nltk.download('stopwords')

def clean_html(raw_html):
    # Parse the raw HTML content using BeautifulSoup
    soup = BeautifulSoup(raw_html, 'html.parser')

    # Extract text content from the HTML
    text_content = soup.get_text()

    # Remove newline characters (\n)
    cleaned_text = text_content.replace('\n', '')

    # Remove Unicode characters, non-alphanumeric characters, and consecutive whitespace
    cleaned_text = re.sub(r'[^\w\s\d]+', ' ', cleaned_text.encode('ascii', 'ignore').decode())

    return cleaned_text.strip()  # Remove leading and trailing whitespace

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove punctuation
    tokens = [word for word in tokens if word not in punctuation]
    
    # Remove stopwords and numeric values
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and not word.isdigit()]
    
    return tokens

def clean_json_data(input_file_paths, output_file_path):
    # Open text file for writing preprocessed text content
    with open(output_file_path, 'w') as txt_file:
        # Process each input JSON file
        for input_file_path in input_file_paths:
            # Read JSON data from input file
            with open(input_file_path, 'r') as f:
                data = json.load(f)

            # Preprocess HTML content for each entry
            for entry in data:
                cleaned_text = clean_html(entry['raw_html'])
                
                # Preprocess cleaned text
                preprocessed_tokens = preprocess_text(cleaned_text)
                
                # Write preprocessed text content to text file
                for token in preprocessed_tokens:
                    txt_file.write(token + '\n')

# List of input JSON file paths
input_file_paths = ['scraped_data15.json', 'scraped_data16.json', 'scraped_data17.json', 'scraped_data18.json', 'scraped_data20.json']
# Path to output text file
output_file_path = 'non_fintech_data.txt'

# Clean JSON data, preprocess text, and write to a text file
clean_json_data(input_file_paths, output_file_path)
