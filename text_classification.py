import itertools
import nltk
import random


from unidecode import unidecode

from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        self._votes = []

    def classify(self, featureset: list) -> list:
        """Runs the given featureset through all classifiers. Returns the mode of results."""
        self._votes = [c.classify(featureset) for c in self._classifiers]
        return mode(self._votes)

    def get_confidence_of_latest_vote(self) -> float:
        return self._votes.count(mode(self._votes)) / len(self._votes)


def main():

    with open('english_dictionary.txt') as dict_file:
        english_words = {word.strip().lower() for word in dict_file}

    documents = []
    short_pos = unidecode(open('positive.txt').read())
    short_neg = unidecode(open('negative.txt').read())

    for review in short_pos.split('\n'):
        documents.append((review, 'pos'))
    for review in short_neg.split('\n'):
        documents.append((review, 'neg'))
    short_pos_words = word_tokenize(short_pos)
    short_neg_words = word_tokenize(short_neg)

    all_words = nltk.FreqDist(w.lower() for w in itertools.chain(short_pos_words, short_neg_words))

    word_features = [w[0] for w in all_words.most_common(5000)
                     if len(w[0]) > 2 and w[0] in english_words]

    featuresets = [(find_features(rev, word_features), category)
                   for (rev, category) in documents]

    random.shuffle(featuresets)

    training_set = featuresets[:10000]
    testing_set = featuresets[10000:]

    algorithm_list = [MultinomialNB, BernoulliNB, LogisticRegression, SGDClassifier, SVC,
                      LinearSVC, NuSVC]
    trained_algorithm_list = create_and_train_algorithms(algorithm_list, training_set)
    test_algorithm_accuracy(trained_algorithm_list, testing_set)

    vote_classifier = VoteClassifier(*trained_algorithm_list)
    print(f"{'VoteClassifier':<20} {nltk.classify.accuracy(vote_classifier, testing_set)}")
    for i in range(10):
        print(f"Classification: {vote_classifier.classify(testing_set[i][0])} Confidence: {vote_classifier.get_confidence_of_latest_vote()}")


def create_and_train_algorithms(algorithm_list, training_set):
    """Takes a list of machine learning algorithm constructor functions. Instantiates each of
    the algorithms, trains them with the given training set, and returns the list of trained
    algorithms."""
    training_list = [SklearnClassifier(algorithm()) for algorithm in algorithm_list]
    for algorithm in training_list:
        algorithm.train(training_set)
    return training_list


def test_algorithm_accuracy(algorithm_list, testing_set):
    """Tests and prints the accuracy of each algorithm in the list."""
    for algorithm in algorithm_list:
        print(f"{algorithm._clf.__class__.__name__:<20} {nltk.classify.accuracy(algorithm, testing_set)}")


def find_features(document, word_features):
    """Returns a dictionary of word-bool pairs. The words are taken from word_features, and
    the bool tells whether or not that word exists in the given document."""
    doc_words = set(word_tokenize(document))
    return {word: (word in doc_words) for word in word_features}


if __name__ == "__main__":
    main()
