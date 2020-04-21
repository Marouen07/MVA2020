import argparse
import re
from nltk.tree import Tree
from sklearn.model_selection import train_test_split
from collections import defaultdict
from nltk.grammar import Nonterminal
import numpy as np
from itertools import product
from tqdm import tqdm
import pickle

from PCFG import PCFG
from OOV import OOV
#Parsing Arguments

# Globals
UNARY_JOIN_CHAR = '+'
UNK_TOKEN = '<UNK>'
NPP_TOKEN = '<NPP>'
START_LABEL = 'SENT'
seperator = ' '


def tree_to_sentence(line):
    """Get the sequence of token from a treebank sample"""
    INNER_PARENTHESIS = re.compile(r'\(([^()]+)\)')
    matchs = INNER_PARENTHESIS.findall(line)
    regex_cleaner = re.compile(r'(-)\w+')
    term_tags = []
    for match in matchs:
        term = match[match.find(' ')+1:]
        term_tags.append(term.strip())
    return term_tags

def sequoia_format(tree):
    return (' '.join(str(tree).split()))


def _build_tree(words, back, i, j, node):
    """
    Recursively build the tree from the back table.
    """
    tree = Tree(node.symbol(), children=[])
    if (i, j) == (j - 1, j):
        tree.append(words[j - 1])
        return tree
    else:
        if (i, j, node) in back.keys():
            k, b, c = back[i, j, node]
            tree.append(_build_tree(words, back, i, k, b))
            tree.append(_build_tree(words, back, k, j, c))
            return tree
        else:
            return tree


def probabilitic_CKY(words, oov, pcfg):
    """
    Probabilistic CYK algorithm. Returns the most likely tree corresponding
    to a given sentence using the learned PCFG.

    Args:
    - words (list) : list of the words composing the sentence

    Returns:
    - parsed (bool) : True if the sentence was successfully parsed
    - tree (nltk.Tree) : Tree corresponding to the parsing, if sentence was succesfully parsed.
    """
    proba_table = defaultdict(float)
    table = defaultdict(list)
    back = dict()

    if len(words) == 1:
        if words[0] in pcfg.vocab:
            word_in_voc = words[0]
        else:
            print("original word: " + words[0])
            word_in_voc = oov.oov_close_word(words[0], None, None)
            print("Corrected word: " + word_in_voc)
        for A, proba in pcfg.rev_lexicon_proba[word_in_voc].items():
            proba_table[(0, 1, A)] = proba

    for j in range(1, len(words) + 1):

        if words[j - 1] in pcfg.vocab:
            word_in_voc = words[j - 1]
        else:
            print("original word: " + words[j - 1])
            if j == 1:
                prev_word = None
            else:
                prev_word = words[j - 2]
            if j == len(words):
                next_word = None
            else:
                next_word = words[j]
            word_in_voc = oov.oov_close_word(words[j - 1], prev_word, next_word)
            print("Corrected word: " + word_in_voc)

        for A, proba in pcfg.rev_lexicon_proba[word_in_voc].items():
            proba_table[(j - 1, j, A)] = proba
            table[(j - 1, j)].append(A)

        for i in range(j - 2, -1, -1):
            for k in range(i + 1, j):
                current_non_term = product(table[(i, k)], table[(k, j)])
                for B, C in current_non_term:
                    for A, proba in pcfg.rev_grammar_proba[(B, C)].items():
                        cumulated_proba = proba + proba_table[(i, k, B)] + proba_table[(k, j, C)]
                        if (proba_table[i, j, A] < cumulated_proba):
                            proba_table[(i, j, A)] = cumulated_proba
                            table[(i, j)].append(A)
                            back[(i, j, A)] = (k, B, C)

    parsed = False
    maxi = -np.inf
    for n1, n2, A in proba_table.keys():
        if (n1, n2, A) == (0, len(words), A):
            if proba_table[0, len(words), A] > maxi:
                h_prob = proba_table[0, len(words), A]
                top_node = A
                parsed = True

    if parsed:
        tree = _build_tree(words, back, 0, len(words), top_node)
        tree.set_label(START_LABEL)
        tree.un_chomsky_normal_form(unaryChar=UNARY_JOIN_CHAR)
    else:
        tree = None

    return parsed, tree

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='run script for the Parser')
    p.add_argument('--input', type=str, required=False,default='test.txt', help='Input file with text to be parsed')
    p.add_argument('--sequoia', type=str, required=False, default="data/sequoia-corpus+fct.mrg_strict",
                   help='sequoia dataset file directory')
    p.add_argument('--polyglot', type=str, required=False, default='./data/polyglot-fr.pkl',
                   help='Polyglot dataset file directory')
    p.add_argument('--output', type=str, required=False, default='evaluation_data.parser_output', help='Output file ')

    args = p.parse_args()

    # loading data
    print('loading data')
    file_corpus = open(args.sequoia, "r", encoding='utf-8')
    corpus = []
    for line in file_corpus.readlines():
        # we'll use unidecode to get rid of accents
        line = line.rstrip('\n')
        corpus.append(line)
    file_corpus.close()

    corpus_train, corpus_test = train_test_split(corpus, test_size=0.1, shuffle=False)
    corpus_train, corpus_val = train_test_split(corpus_train, test_size=0.1 / 0.9, shuffle=False)

    # Hlobal Vars
    UNARY_JOIN_CHAR = '+'
    UNK_TOKEN = '<UNK>'
    NPP_TOKEN = '<NPP>'
    START_LABEL = 'SENT'
    seperator = ' '

    pcfg = PCFG()
    pcfg.create_pcfg(corpus_train)
    oov = OOV(pcfg)
    oov.get_polyglot()
    train_sequence_list = [tree_to_sentence(seq) for seq in corpus_train]
    oov.create_bigram(train_sequence_list)
    seqs = []
    input_file = open(args.input, "r")


    for line in tqdm(input_file.readlines()):
        print(line)
        line = line.rstrip('\n')
        parsable, tree = probabilitic_CKY(line.split(),oov,pcfg)
        if parsable:
            seq_format = sequoia_format(tree)

            seqs += [seq_format + '\n']
            file = open(args.output, "a")
            file.write(seqs[-1])
            file.close()
        else:
            seqs += [line + '\n']
            print(' sentence \n' + line + ' \n could not be parsed \n')
            file = open(args.output, "a")
            file.write(seqs[-1])
            file.close()
    input_file.close()


