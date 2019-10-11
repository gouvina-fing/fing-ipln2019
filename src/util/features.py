import re
import math
import numpy as np

def read_dictionary(filename):
    with open(filename) as dictionary:
        return {word.rstrip('\n') for word in dictionary if word.rstrip('\n')}

# CONSTANTS

# Lista de símbolos de diálogo posibles. Parecen todos iguales, pero son distintos
# (algunos son hyphen, otros dashes, otros minus sign, viñetas, etc.)
dialogue_punctuation = ['-', '—', '–', '―', '‒', '‐', '−', '­', '‑', '⁃', '֊', '˗', '⁻', '⏤', '─', '➖']

url_regex = re.compile(
    r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)',
    re.IGNORECASE
)

question_answer_regex = re.compile(
    r"""
    ¿+ [^\?]+ \?+ # question
    [^¿\?]* [\w\d] # answer
    """, re.VERBOSE
)

word_or_number_regex = re.compile(r'\b\w+\b')
capslock_word_regex = re.compile(r'\b[0-9A-Z_]+\b')

keywords = read_dictionary('src/util/dictionaries/keywords.dic')
keywords = [x.lower() for x in keywords]

sexual_words = read_dictionary('src/util/dictionaries/sexual.dic')
sexual_words = [x.lower() for x in sexual_words]

animals = read_dictionary('src/util/dictionaries/animales.dic')
animals = [x.lower() for x in animals]

hasthag_regex = re.compile(r'(\B#\w+)')

exclamation_regex = re.compile(r'(¡\w*!|!\s+)')

# MAIN FUNCTIONS
def get_features(tweets):
    return np.vectorize(extract_features)(tweets)

# AUXILIARY FUNCTIONS

def extract_features(tweet):
    # Eliminate linebreaks
    tweet = re.sub('\n', ' ', tweet)

    downcased_tweet = tweet.lower()

    features = {}

    features['starts_with_dialogue'] = starts_with_dialogue(tweet)

    features['number_of_urls'] = number_of_regex_occurrences(url_regex, tweet)
    features['number_of_exclamations'] = number_of_regex_occurrences(exclamation_regex, tweet)
    features['number_of_hashtags'] = number_of_regex_occurrences(hasthag_regex, tweet)
    features['number_of_question_answers'] = number_of_regex_occurrences(question_answer_regex, tweet)

    features['number_of_keywords'] = number_of_word_ocurrences(keywords, tweet)
    features['number_of_animals'] = number_of_word_ocurrences(animals, tweet)
    features['number_of_sexual_words'] = number_of_word_ocurrences(sexual_words, tweet)

    features['capslock_ratio'] = capslock_ratio(tweet)
    
    return features

############################

# Determines if tweet starts with any dialogue symbol
def starts_with_dialogue(tweet):
    for punctuation in dialogue_punctuation:
        if tweet.startswith(punctuation):
            return 1
    return 0

# Counts the number of groups when finding by a specific regex in the tweet
def number_of_regex_occurrences(regex, tweet):
    return len(re.findall(regex, tweet))

# Counts the number of occurrences of a specific list in the tweet
def number_of_word_ocurrences(words, tweet):
    number_of_occurrences = 0
    tweet = tweet.lower()
    words = tweet.split(' ')
    for word in words:
        if word in words:
            number_of_occurrences += 1
    return number_of_occurrences / math.sqrt(len(words))

# Cuenta la cantidad de palabras totalmente en mayúsculas, dividido la cantidad de palabras del tweet.
def capslock_ratio(tweet):
    number_of_word_or_numbers = len(re.findall(word_or_number_regex, tweet))
    number_of_capslock_words = len(re.findall(capslock_word_regex, tweet))

    if number_of_word_or_numbers == 0:
        return 0
    else:
        return number_of_capslock_words / number_of_word_or_numbers
