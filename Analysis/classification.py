'''
An implementation of the Twitter analysis tool with the help of tutorial
http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/

sad.txt contains > 500 negative tweets
happy.txt contains > 500 positive tweets

sad_test.txt contains 20 random tweets
happy_test.txt contains 20 random tweets

'''
import nltk
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk import pos_tag, word_tokenize
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

def cloud(set):
    spamcloud = WordCloud().generate(set)
    plt.figure()
    plt.imshow(spamcloud)
    plt.axis("off")
    plt.show()

    return spamcloud

def read_tweets(fname, t_type):
    tweets = []
    f = open(fname, 'r')
    line = f.readline()
    while line != '':
        tweets.append([line, t_type])
        line = f.readline()
    f.close()
    return tweets

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
      features['contains(%s)' % word] = (word in document_words)
    return features


def classify_tweet(tweet):
    return \
        classifier.classify(extract_features(nltk.word_tokenize(tweet)))


# read in postive and negative training tweets
pos_tweets = read_tweets('happy.txt', 'positive')
neg_tweets = read_tweets('sad.txt', 'negative')

# filter away words that are less than 3 letters
tweets = []
for (words, sentiment) in pos_tweets + neg_tweets:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    tweets.append((words_filtered, sentiment))

#print(tweets[0:])

# extract the word features out from the training data
word_features = get_word_features(\
                    get_words_in_tweets(tweets))


# get the training set and train the Naive Bayes Classifier
training_set = nltk.classify.util.apply_features(extract_features, tweets)
classifier = NaiveBayesClassifier.train(training_set)


# read in the test tweets and check accuracy
# to add your own test tweets, add them in the respective files
test_tweets = read_tweets('happy_test.txt', 'positive')
test_tweets.extend(read_tweets('sad_test.txt', 'negative'))
total = accuracy = float(len(test_tweets))

for tweet in test_tweets:
    if classify_tweet(tweet[0]) != tweet[1]:
        accuracy -= 1

print("----------------------Most Informative Features------------------------------")
print (classifier.show_most_informative_features(32))


print("----------------------Test Data sentiment analysis------------------------------")
print('Total accuracy: %f%% (%d/20).' % (accuracy / total * 100, accuracy))

print("----------------------Tweets From Twitter------------------------------")

mantweet1 = 'im the type to miss people even after they hurt me deeply. im too much of a kind soul and i hate it'
print("@user : ",mantweet1)
print("----------------------Tokenizing the TWEET & Classification------------------------------")
tweet1_tagged=pos_tag(word_tokenize(mantweet1))
print(tweet1_tagged)

print ("sentiment analysis result:" ,classifier.classify(extract_features(mantweet1.split())))

print("------------------------------------------------------------------------")
mantweet2 = '''
We're going to use American steel, we're going to use American labor we are going to come first in all deals.'
'''
print("@POTUS : ",mantweet2)
print("----------------------Tokenizing the TWEET & Classification------------------------------")
print(pos_tag(word_tokenize(mantweet2)))

print ("sentiment analysis result:" ,classifier.classify(extract_features(mantweet2.split())))



#create word cloud for posotive and negative tweets from training set
with open('happy.txt', 'r') as myfile:
    happy=myfile.read().replace('\n', '')

#create a word cloud
charth=cloud(happy)

df_spam = pd.DataFrame()
df_spam['words'] = charth.words_.keys()
df_spam['frequencies'] = charth.words_.values()
plt.figure(figsize=(12, 6))
sns.barplot(x='words', y='frequencies', data=df_spam.sort_values(by=['frequencies'], ascending=[0]).head(20))

with open('sad.txt', 'r') as myfile:
    sad=myfile.read().replace('\n', '')

cloud(sad)
