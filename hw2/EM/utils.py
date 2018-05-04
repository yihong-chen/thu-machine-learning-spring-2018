import re
import nltk
import numpy as np


def preprocess(data):
    pre = []
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords.update(['would', 'subject', 're', 'don', 'jan', 'feb', 'mar', 'apr', 'may', 'june', 'july', 'aug', 'sep', 'oct', 'nov', 'dec'])
    # Regex pattern for general special characters
    #special_char_pattern = re.compile(r'[\|=-\[\]\'\":;,\.\<\>\\\/\?_\(\)!$%^&*,]', re.IGNORECASE & re.UNICODE)
    special_char_pattern = re.compile(r"[^\w']|_")
    # Regex pattern for whole-word numbers
    number_pattern = re.compile(r'\b\d+\b')

    # Loop through data do the preprocessing
    for j in range(0, len(data)):
        lines = data[j].lower().split("\n")
        for i in range(0, len(lines)):
            lines[i] = number_pattern.sub(' ', lines[i])
            lines[i] = special_char_pattern.sub(' ', lines[i])
            # Remove short words
            lines[i] = ' '.join([w for w in lines[i].split() if len(w) > 2])
            # Remove stopwords
            lines[i] = ' '.join([w for w in lines[i].split() if w not in stopwords])
            # Stem the words
            lines[i] = ' '.join([nltk.stem.snowball.SnowballStemmer("english").stem(w) for w in lines[i].split()])
        doc = " ".join(lines)
        pre.append(doc)
    return pre


def visual_frequent_words(log_mu, vocabulary, topN=20):
    """
    args:
        log_mu: numpy array of shape (num_words, num_topics)
    """
    num_words, num_mixtures = log_mu.shape
    top_words_idx = np.argpartition(log_mu, -topN, axis=0)[-topN:]
    for topic_id in range(num_mixtures):
        print('-' * 80)
        print('Frequent words of topic {}:'.format(topic_id))
        words = [vocabulary[word_idx] for word_idx in top_words_idx[topic_id]]
        print(' '.join(words))