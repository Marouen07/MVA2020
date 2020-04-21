import pickle
import numpy as np
import re


# Globals
UNARY_JOIN_CHAR = '+'
UNK_TOKEN = '<UNK>'
NPP_TOKEN = '<NPP>'
START_LABEL = 'SENT'
seperator = ' '


def damereau_levenstein_distance(w1, w2):
    """
    Compute the Levenshtein damereau distance between two strings
    ------------------------------------------------------------------
    Input:
    - word1, word2 (str)
    Returns:
    - distance (int)
    """
    word1, word2 = w1.lower(), w2.lower()
    n2 = len(word2)
    distance = np.zeros((3, n2 + 1))
    distance[0, :] = np.arange(n2 + 1)
    distance[1, 0] = 1
    for j in range(1, n2 + 1):
        diff_last_letters = word1[0] != word2[j - 1]
        distance[1, j] = min([distance[0][j] + 1, distance[1][j - 1] + 1, distance[0][j - 1] + diff_last_letters])

    for i in range(2, len(word1) + 1):
        distance[2][0] = i
        for j in range(1, n2 + 1):
            diff_last_letters = word1[i - 1] != word2[j - 1]
            distance[2, j] = min([distance[1][j] + 1, distance[2][j - 1] + 1, distance[1][j - 1] + diff_last_letters])
            # damereau for considering character swaps as well
            if j > 1:
                if (word1[i - 1] == word2[j - 2]) and (word1[i - 2] == word2[j - 1]):
                    distance[2, j] = min(distance[2, j], distance[0, j - 2] + 1)
        distance[0, :] = distance[1, :]
        distance[1, :] = distance[2, :]

    return int(distance[2][n2])



class OOV(object):
    '''Module to Handle OOV words.
    '''

    def __init__(self, pcfg):
        '''
            Input:
        ---------------------------

                pcfg:   Probabilistic Context free Grammar Object *

        ---------------------------
        '''
        self.pcfg = pcfg
        self.polyglot_words = None
        self.embeddings = None
        self.word2idx = None  # word2idx for polyglot words
        self.idx2word = None  # idx2word for polyglot words

    def get_polyglot(self, polyglot_data_path='./data/polyglot-fr.pkl'):

        '''Get the word embeddings from a pkl file
        -------------------------------
            Input:
                polyglot_data_path: the pickle file containing polyglot embeddings
        -------------------------------
        '''

        polyglot_data_path = './data/polyglot-fr.pkl'
        # Polyglot data
        with open(polyglot_data_path, 'rb') as file:
            self.polyglot_words, self.embeddings = pickle.load(
                file, encoding='iso-8859-1')
        self.word2idx = {w: i for (i, w) in enumerate(self.polyglot_words)}
        self.idx2word = dict(enumerate(self.polyglot_words))
        # self.vocab_polygot_2idx={w:word2idx[w] for w in vocab if w in self.polyglot_words}

    def case_sub(self, word, vocabulary):
        """Case Normalising Procedures in case the word is not in
        Vocab.
        we only keep the match with the lowest index if it exists.
        """
        w = word[:]
        lower = (vocabulary.get(w.lower(), -1), w.lower())
        upper = (vocabulary.get(w.upper(), -1), w.upper())
        title = (vocabulary.get(w.title(), -1), w.title())
        index, w = sorted([lower, upper, title])[0]
        if index == -1:
            return word
        else:
            return w

    def normalize(self, word, vocabulary):
        '''The whole normalizer including the case of digits.
        ------------------------------
            Input:
                word: the word to normalize
        ------------------------------
            Return:
                the normalized word. We can not ensure that the normalize word will be in the embedding vocabulary.
        '''
        DIGITS = re.compile("[0-9]", re.UNICODE)
        if word not in vocabulary:
            word = DIGITS.sub("#", word)
        if word not in vocabulary:
            word = self.case_sub(word, vocabulary)
        return word

    def cosine_similarity(self, word1, word2):

        '''Compute cosine similarity between two words
        -------------------------------
            Input:
                word1: the first word
                word2: the second word
        -------------------------------
            Return:
                similarity between two words' embeddings
        '''

        embed1 = self.embeddings[self.word2idx[word1]]
        embed2 = self.embeddings[self.word2idx[word2]]
        distance = np.dot(np.reshape(embed1, (1, -1)), np.reshape(embed2, (-1, 1))) / (
                np.linalg.norm(embed1) * np.linalg.norm(embed2))
        return distance

    def Nearest_Levenstein(self, word, dist=2):

        '''Generate candidates which are within distance k to the unseen word
        --------------------------------
            Input:
                word: the unseen word
                dist: the tolerance of distance, defaut: 2
        --------------------------------
            Returns:

                corrections : possible corrections
        '''

        corrections = []
        for vocab_word in self.pcfg.vocab.keys():
            if damereau_levenstein_distance(word, vocab_word) <= dist:
                corrections.append(vocab_word)
        return corrections

    def create_bigram(self, train_sentences):

        '''
        constructs the Bigram  matrix.
        ------------------------------
            Input:
                train_sentences: list of training sentences
        ------------------------------
            Updates self.bigram
        '''
        # Creating a mapping of our Vocab
        l = len(self.pcfg.word2idx)
        self.bigram = np.zeros((l + 1, l + 1))
        for words in train_sentences:
            self.bigram[l, self.pcfg.word2idx[words[0]]] += 1
            for i in range(len(words) - 1):
                self.bigram[self.pcfg.word2idx[words[i]], self.pcfg.word2idx[words[i + 1]]] += 1
            self.bigram[self.pcfg.word2idx[words[-1]], l] += 1
        self.bigram = self.bigram / np.sum(self.bigram, axis=1)

    def get_bigram_score(self, prev_word, next_word, corrections):

        '''Calcualtes Bigram probability of each correction which is considered
        as a score to choose the right one
        ------------------------------
            Input:
                prev_word: the previous word of the unseen word in the sentence
                next_word: the next word of the unseen word in the sentence
                corrections: the list of possible words to compute scores
        ------------------------------
            Return:
                a list of probabilities corresponding to each condidate
        '''
        l = self.bigram.shape[0] - 1
        lefts = []
        rights = []
        if prev_word is None:
            for correction in corrections:
                lefts.append(self.bigram[l, self.pcfg.word2idx[correction]])
        else:
            if prev_word in self.pcfg.word2idx.keys():
                for correction in corrections:
                    lefts.append(self.bigram[self.pcfg.word2idx[prev_word], self.pcfg.word2idx[correction]])
            else:
                lefts = [1.] * len(corrections)
        if next_word is None:
            for correction in corrections:
                rights.append(self.bigram[self.pcfg.word2idx[correction], l])
        else:
            if next_word in self.pcfg.word2idx.keys():
                for correction in corrections:
                    rights.append(self.bigram[self.pcfg.word2idx[correction], self.pcfg.word2idx[next_word]])
            else:
                rights = [1.] * len(corrections)

        probas = np.multiply(lefts, rights)
        if sum(probas) == len(corrections):
            for i, correction in enumerate(corrections):
                probas[i] = self.pcfg.vocab[correction]
            probas = probas / sum(probas)
        return probas

    def oov_close_word(self, word, prev_word, next_word, dist=2, lamda=1000):

        '''Assign an unique similar token to the unseen word
        --------------------------------
            Input:
                word: the unseen word
                prev_word: the previous word
                next_word: the next word
                dist: the tolerance of distance
                lamda: the hyperparameter to adjust the importance of similarity score and bigram score
        --------------------------------
            Return:
                Closest Possible Word
        '''
        # Simple Cases Tests
        if word in self.pcfg.vocab:
            return word
        else:
            w = self.normalize(word, self.pcfg.vocab)

        if w in self.pcfg.vocab:
            return w
            # More advanced tests
        if w not in self.word2idx:
            w = self.normalize(word, self.word2idx)
        if w in self.word2idx.keys():
            all_tokens = list(self.pcfg.word2idx.keys())
            simis = np.zeros(len(all_tokens))
            for token in all_tokens:
                t = self.normalize(token, self.word2idx)
                if t in self.word2idx.keys():
                    simis[self.pcfg.word2idx[token]] = self.cosine_similarity(w, t)
            candidates = self.Nearest_Levenstein(word, dist)
            if len(candidates) == 0:
                return self.pcfg.idx2word[np.argmax(simis)]
            else:
                bigram_probas = self.get_bigram_score(prev_word, next_word, candidates)
                for i, candidate in enumerate(candidates):
                    idx = self.pcfg.word2idx[candidate]
                    simis[idx] += lamda * bigram_probas[i]
                return self.pcfg.idx2word[np.argmax(simis)]
        else:
            candidates = self.Nearest_Levenstein(word, dist)
            if len(candidates) == 0:
                #print('we"re here')
                return UNK_TOKEN
                # candidates = list(self.pcfg.word2idx.keys())
            bigram_probas = self.get_bigram_score(prev_word, next_word, candidates)
            best_match = np.argmax(bigram_probas)

            return candidates[best_match]