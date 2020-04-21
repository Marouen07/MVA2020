import re
from nltk.tree import Tree
from collections import defaultdict
from nltk.grammar import Nonterminal
import numpy as np

# Globals
UNARY_JOIN_CHAR = '+'
UNK_TOKEN = '<UNK>'
NPP_TOKEN = '<NPP>'
START_LABEL = 'SENT'
seperator = ' '


def create_tree(sentence):
    """Creating nltk Parsing Tree from Sequoia text file format
    -----------------------------
    input :
    sentence: Line Of text from Sequoia dataset
    Returns:
    tree: nltk.tree object
    """

    tree = Tree.fromstring(sentence, remove_empty_top_bracketing=True)
    clean(tree)
    tree.chomsky_normal_form()
    # tree.collapse_unary(collapseRoot=True, collapsePOS=True)
    return tree

def clean(tree):
    """
    We'll get rid of functional labels
    as instructed in the TD
    --------------------------------------------
    Input:
    tree :  nltk.Tree object
    """
    tree.set_label(re.sub(r'(-)\w+', '', tree.label()))
    for child in tree:
        if isinstance(child, Tree):
            clean(child)


class PCFG(object):

    def __init__(self):
        self.lexicon_proba = defaultdict(float)
        self.grammar_proba = defaultdict(float)
        self.lhs_grammar = defaultdict(float)
        self.lhs_lexicon = defaultdict(float)
        self.vocab = defaultdict(float)
        self.rev_grammar_proba = defaultdict(dict)
        self.rev_lexicon_proba = defaultdict(dict)


    def Normalize(self, counts, totals):
        """
        convert counts into probabilities
        """
        for rule, count in counts.items():
            counts[rule] = count / totals[rule[0]]
        return counts

    def extract_prods(self, corpus):
        """Extracting Production rules from corpus"""
        productions = []
        for sentence in corpus:
            tree = create_tree(sentence)
            productions.extend(tree.productions())
        return productions

    def add_UNK(self):
        """
        For OOV words that we could not identify, use the probabilities of the
        most likely POS tags.
        """
        total_count = sum(self.lhs_lexicon.values())
        for tag in self.lhs_lexicon.keys():
            self.lexicon_proba[(tag, '<UNK>')] = self.lhs_lexicon[tag] / total_count

    def create_pcfg(self, corpus_train):
        """
        Create PCFG from Treebank Train Corpus

        """
        # extracting Productions from corpus using  nltk
        productions = self.extract_prods(corpus_train)
        for rule in productions:
            if rule.is_lexical():

                # lexicon keys : (Tag,word)
                wd = seperator.join(rule.rhs())
                self.lexicon_proba[(rule.lhs(), wd)] += 1
                self.lhs_lexicon[rule.lhs()] += 1
                self.vocab[str(wd)] += 1

            else:
                # grammar :
                self.grammar_proba[(rule.lhs(), rule.rhs())] += 1
                self.lhs_grammar[rule.lhs()] += 1


        # Turning Counts to Probabilities
        self.grammar_proba = self.Normalize(self.grammar_proba, self.lhs_grammar)
        self.lexicon_proba = self.Normalize(self.lexicon_proba, self.lhs_lexicon)
        self.word2idx = word2idx = {w: i for (i, w) in enumerate(self.vocab.keys())}
        self.idx2word = dict(enumerate(self.vocab.keys()))
        # Adding Tags for OOVs
        # add_NPP(self.lexicon_proba, self.lhs_lexicon)
        self.add_UNK()

        # mapping gammar rules to their probabilities
        for rule, proba in self.grammar_proba.items():
            self.rev_grammar_proba[rule[1]][rule[0]] = proba
        # mapping lexicons to their probabilities
        for rule, proba in self.lexicon_proba.items():
            self.rev_lexicon_proba[rule[1]][rule[0]] = proba