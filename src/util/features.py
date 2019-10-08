import re
import numpy as np

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

#multiple_spaces_regex = re.compile(r' +')

#retweet_regex = re.compile(r'^RT @\w+: ')

#hasthag_regex = re.compile(r'(\B#\w+)')

#tag_regex = re.compile(r'(\B@\w+)')


# MAIN FUNCTIONS
def get_features(tweets):
    return np.vectorize(extract_features)(tweets)
    #return (lambda tweet: extract_features(tweet))(tweets)

# AUXILIARY FUNCTIONS

def extract_features(tweet):
    features = {}

    features['contains_dialogue'] = contains_dialogue(tweet)
    features['number_of_urls'] = number_of_urls(tweet)
    features['number_of_question_answers'] = number_of_question_answers(tweet)

    return features

# 
def contains_dialogue(tweet):
    for punctuation in dialogue_punctuation:
        if tweet.startswith(punctuation):
            return 1
    return 0

# 
def number_of_urls(tweet):
    return len(re.findall(url_regex, tweet))

#
def number_of_question_answers(tweet):
    # return len(re.findall(question_answer_regex, tweet))
    return question_answer_regex.subn('', tweet)[1]