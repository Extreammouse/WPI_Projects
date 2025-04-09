#task1
import pandas as pd
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from tabulate import tabulate

def download_nltk_resources():
    """Download required NLTK resources"""
    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")

download_nltk_resources()

def preprocess_text(text):
    """Preprocess text with tokenization, stopword removal, and lemmatization"""
    if not isinstance(text, str):
        text = str(text)
    
    tokens = [word.lower() for word in text.split()]
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {'would', 'said', 'one', 'also', 'get', 'like', 'new', 
                        'said', 'says', 'just', 'will', 'time', 'year'}
    stop_words.update(custom_stopwords)
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words and len(token) > 2]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

def create_wordcloud(word_freq, title, filename):
    """Generate and save a word cloud from word frequencies"""
    wordcloud = WordCloud(
        width=1200,
        height=800,
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=20, pad=20)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Word cloud saved to {filename}")

def process_file_contents(df):
    """Process all articles in a dataframe and return word frequencies"""
    all_tokens = []
    for text in df['text']:
        tokens = preprocess_text(text)
        all_tokens.extend(tokens)
    return dict(Counter(all_tokens).most_common(100))

def display_top_words(freq_dict, title):
    """Display the top 100 words and their frequencies in a table format"""
    print(f"\n{title} (Top 100 Words):")
    table_data = [(rank+1, word, count) for rank, (word, count) in enumerate(freq_dict.items())]
    headers = ["Rank", "Word", "Frequency"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def analyze_news(true_csv_path, fake_csv_path):
    """Analyze both real and fake news articles with visualizations"""
    try:
        print("Reading CSV files...")
        true_df = pd.read_csv(true_csv_path)
        fake_df = pd.read_csv(fake_csv_path)
        
        print(f"Analyzing {len(true_df)} real articles and {len(fake_df)} fake articles...")
        
        print("\nProcessing real news articles...")
        real_freq = process_file_contents(true_df)
        create_wordcloud(real_freq, "Real News Word Cloud", "real_wordcloud.png")
        display_top_words(real_freq, "Real News Articles")
        
        print("\nProcessing fake news articles...")
        fake_freq = process_file_contents(fake_df)
        create_wordcloud(fake_freq, "Fake News Word Cloud", "fake_wordcloud.png")
        display_top_words(fake_freq, "Fake News Articles")        
        print("\nProcessing combined news articles...")
        combined_df = pd.concat([true_df, fake_df])
        all_freq = process_file_contents(combined_df)
        create_wordcloud(all_freq, "Combined News Word Cloud", "combined_wordcloud.png")
        display_top_words(all_freq, "Combined News Articles")        
        print("\nComparing most distinct words between real and fake news...")
        common_words = set(real_freq.keys()) & set(fake_freq.keys())
        
        difference_ratios = []
        for word in common_words:
            real_count = real_freq[word]
            fake_count = fake_freq[word]
            real_relative = real_count / sum(real_freq.values())
            fake_relative = fake_count / sum(fake_freq.values())
            difference_ratio = abs(real_relative - fake_relative) / max(real_relative, fake_relative)
            difference_ratios.append((word, difference_ratio, real_count, fake_count))
        
        difference_ratios.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop 20 Most Distinct Words (appearing in both real and fake news):")
        distinct_headers = ["Word", "Difference Ratio", "Real News Count", "Fake News Count"]
        distinct_data = [(word, f"{ratio:.2f}", real_count, fake_count) 
                        for word, ratio, real_count, fake_count in difference_ratios[:100]]
        print(tabulate(distinct_data, headers=distinct_headers, tablefmt="grid"))
        
        return all_freq, real_freq, fake_freq
    
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}, {}, {}

if __name__ == "__main__":
    true_csv = "true.csv"
    fake_csv = "fake.csv"
    print("Starting news analysis...")
    all_freq, real_freq, fake_freq = analyze_news(true_csv, fake_csv)