import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import nltk
import warnings
warnings.filterwarnings('ignore')

def download_nltk_resources():
    print("Downloading required NLTK resources...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        print("NLTK resources downloaded successfully.")
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
        print("You might need to manually download the resources.")
        print("Run the following in a Python interpreter:")
        print(">>> import nltk")
        print(">>> nltk.download('punkt')")

print("Loading Amazon Fine Food Reviews dataset...")
data = pd.read_csv("amazon-fine-foods/Reviews.csv")

def perform_eda(data):
    print("\n--- Exploratory Data Analysis ---")
    print(f"Dataset shape: {data.shape}")
    print("\nFirst few rows:")
    print(data.head())
    
    print("\nData types:")
    print(data.dtypes)
    
    print("\nMissing values:")
    print(data.isnull().sum())
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x='Score')
    plt.title('Distribution of Review Scores')
    plt.savefig('score_distribution.png')
    plt.close()
    
    data['word_count'] = data['Text'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x='word_count', bins=50)
    plt.title('Distribution of Word Count in Reviews')
    plt.xlim(0, 200)
    plt.savefig('word_count_distribution.png')
    plt.close()
    
    return data

def create_labels(data):
    print("\n--- Creating Labels ---")
    data['sentiment'] = data['Score'].apply(lambda x: 1 if x > 3 else 0)
    
    print("Class distribution:")
    print(data['sentiment'].value_counts())
    
    return data

def balance_dataset(data):
    print("\n--- Balancing Dataset ---")
    neg_count = len(data[data['sentiment'] == 0])
    pos_count = len(data[data['sentiment'] == 1])
    
    if neg_count / pos_count < 0.5 or pos_count / neg_count < 0.5:
        if neg_count < pos_count:
            pos_data = data[data['sentiment'] == 1].sample(n=neg_count * 2, random_state=42)
            neg_data = data[data['sentiment'] == 0]
            balanced_data = pd.concat([pos_data, neg_data])
        else:
            neg_data = data[data['sentiment'] == 0].sample(n=pos_count * 2, random_state=42)
            pos_data = data[data['sentiment'] == 1]
            balanced_data = pd.concat([pos_data, neg_data])
        
        print(f"Balanced dataset shape: {balanced_data.shape}")
        print("New class distribution:")
        print(balanced_data['sentiment'].value_counts())
        return balanced_data
    
    print("Dataset is already balanced enough. No resampling needed.")
    return data

def sample_and_split_data(data, sample_frac=0.15):
    print(f"\n--- Sampling {sample_frac*100}% of Data and Splitting ---")
    sampled_data = data.sample(frac=sample_frac, random_state=42)
    print(f"Sampled dataset shape: {sampled_data.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        sampled_data['Text'], 
        sampled_data['sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=sampled_data['sentiment']
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, sampled_data