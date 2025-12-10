import pandas as pd
import math
import warnings
import os

warnings.filterwarnings('ignore')

# Global storage for training data
texAll_train, lblAll_train = [], []
voc, cat = set(), set()

# ==========================================
# Helper Functions
# ==========================================
def readTrainData(file_path):
    """
    Reads a CSV file and processes the text data.
    Returns:
        texAll: List of tokenized documents (lists of words)
        lbAll: List of labels/categories for each document
        voc_set: Set of all unique words found in the file
        cat_set: Set of all unique categories found
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return [], [], set(), set()

    try:
        # Load CSV, assuming no header, columns are 'category' and 'text'
        df = pd.read_csv(file_path, header=None, names=['category', 'text'])
    except Exception as e:
        print(f"Error reading '{file_path}': {e}")
        return [], [], set(), set()

    texAll = []
    lbAll = []
    voc_set = set()
    cat_set = set()

    for index, row in df.iterrows():
        label = row['category']
        # Simple tokenization: lowercase and split by whitespace
        text = str(row['text']).lower().split()
        
        texAll.append(text)
        lbAll.append(label)
        cat_set.add(label)
        for word in text:
            voc_set.add(word)

    return texAll, lbAll, voc_set, cat_set

def learn_NB_text():
    """
    Trains the Naive Bayes model.
    Calculates Class Priors (P) and Conditional Word Probabilities (Pw).
    """
    # Calculate Class Priors P(Category)
    num_docs = len(lblAll_train)
    category_counts = {}
    for label in lblAll_train:
        category_counts[label] = category_counts.get(label, 0) + 1
    
    # Probability of a document belonging to a category
    P = {c: count / num_docs for c, count in category_counts.items()}

    # Calculate Conditional Probabilities P(Word | Category)
    Pw = {}
    vocab_size = len(voc)
    
    for category in cat:
        Pw[category] = {}
        
        # Get indices of all documents that belong to this category
        indices = [i for i, label in enumerate(lblAll_train) if label == category]
        docs = [texAll_train[i] for i in indices]
        
        # Count frequency of every word in this category
        word_counts = {}
        total_words_in_category = 0
        for doc in docs:
            for word in doc:
                word_counts[word] = word_counts.get(word, 0) + 1
                total_words_in_category += 1
        
        # Calculate probability for every word in the global vocabulary
        # Uses Laplace Smoothing (add-one smoothing) to handle unseen words
        for word in voc:
            count = word_counts.get(word, 0)
            # P(w|c) = (count(w,c) + 1) / (count(all words in c) + |V|)
            Pw[category][word] = (count + 1) / (total_words_in_category + vocab_size)
            
    return Pw, P

def ClassifyNB_text(Pw, P, test_docs, test_labels):
    """
    Predicts categories for test documents and calculates accuracy.
    Uses log-probabilities to avoid arithmetic underflow.
    """
    predictions = []
    
    for doc in test_docs:
        log_probs = {}
        
        # Calculate score for each potential category
        for category in P:
            # Start with Log Prior: log(P(c))
            score = math.log(P[category])
            
            for word in doc:
                # We only care about words we've seen in training (in the vocab)
                if word in voc:
                    # Add Log Likelihood: + log(P(w|c))
                    score += math.log(Pw[category].get(word))
            
            log_probs[category] = score
        
        # The prediction is the category with the highest score (maximum log probability)
        pred = max(log_probs, key=log_probs.get)
        predictions.append(pred)

    # Calculate Accuracy
    if len(test_labels) == 0:
        return 0.0

    correct = sum(1 for i in range(len(predictions)) if predictions[i] == test_labels[i])
    return correct / len(test_labels)

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print("--- Naïve Bayes Text Classification ---")
    
    # Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(script_dir, 'data', 'text_clarification_train.csv')
    test_file = os.path.join(script_dir, 'data', 'text_clarification_test.csv')

    print(f"Loading training data from: {train_file}")
    texAll_train, lblAll_train, voc, cat = readTrainData(train_file)

    if not texAll_train:
        print("Training data failed to load. Exiting.")
        exit()
        
    print(f"  - Vocabulary Size: {len(voc)}")
    print(f"  - Categories Found: {len(cat)}")
    print(f"  - Documents: {len(texAll_train)}")

    print(f"Loading test data from: {test_file}")
    texAll_test, lblAll_test, _, __ = readTrainData(test_file)

    if not texAll_test:
        print("Test data failed to load. Exiting.")
        exit()
    # Run Training and Classification
    print("Training Naive Bayes Model...")
    Pw, P = learn_NB_text()
    
    print("Classifying Test Data...")
    accuracy = ClassifyNB_text(Pw, P, texAll_test, lblAll_test)
    
    print(f"Test Accuracy: {accuracy:.4f}")