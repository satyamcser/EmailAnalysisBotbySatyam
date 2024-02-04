import torch
import transformers
import nltk
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Download pre-trained language model (using gpt2-small)
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("GroNLP/gpt2-small-dutch-embeddings")
model = AutoModelForCausalLM.from_pretrained("GroNLP/gpt2-small-dutch-embeddings")

#define functions for email analysis
def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, max_length=50, return_tensors="pt")
    output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


def analyze_sentiment(email_content):
    # Download the VADER lexicon
    nltk.download('vader_lexicon')

    # Create SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    # Calculate sentiment score
    sentiment_score = sia.polarity_scores(email_content)['compound']
    
    return sentiment_score


def categorize_topic(email_content):
    # Space for topic categorization, we can replace this later with more advanced methods
    if "work" in email_content.lower():
        return "Work"
    elif "personal" in email_content.lower():
        return "Personal"
    else:
        return "Other"

def extract_information(email_content):
    # Space for information extraction, we can replace this later with more advanced methods
    tokens = word_tokenize(email_content)
    filtered_tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
    word_counts = Counter(filtered_tokens)
    # Extract the most frequent words as information
    information = [word for word, count in word_counts.most_common(3)]
    return information

#streamlit app

def main():
    st.title("Email Analysis Assistant with Hugging Face Transformers by Satyam")

    # Download necessary NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')

    user_input_email = st.text_area("Enter the email content:")

    if st.button("Analyze"):
        generated_response = generate_response(user_input_email)
        sentiment_score = analyze_sentiment(user_input_email)
        topic_category = categorize_topic(user_input_email)
        information_extracted = extract_information(user_input_email)

        st.subheader("Generated Response:")
        st.write(generated_response)

        st.subheader("Analysis Results:")
        st.write(f"Sentiment Score: {sentiment_score}")
        st.write(f"Topic Category: {topic_category}")
        st.write(f"Extracted Information: {information_extracted}")

if __name__ == '__main__':
    main()
    

