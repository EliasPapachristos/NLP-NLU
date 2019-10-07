# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'N-Grams'))
	print(os.getcwd())
except:
	pass

#%%
from collections import Counter
import utils


#%%
def bigram_mle(tokens, bigrams):
    """
    provide a dictionary of probabilities for all bigrams in a corpus of text
    the calculation is based on maximum likelihood estimation and does not include
    any smoothing.  A tag '<unk>' has been added for unknown probabilities.
    :param tokens: list
        tokens: list of all tokens in the corpus
    :param bigrams: list
        bigrams: list of all two word tuples in the corpus
    :return: dict
        bg_mle_dict: a dictionary of bigrams:
            key: tuple of two bigram words, in order OR <unk> key
            value: float probability
            
    """
    bg_mle_dict = {}
    bg_mle_dict['<unk>'] = 0.

    token_raw_counts = Counter(tokens)
    bigram_raw_counts = Counter(bigrams)
    for bg in bigram_raw_counts:
        bg_mle_dict[bg] = bigram_raw_counts[bg] / token_raw_counts[bg[0]]
    return bg_mle_dict


#%%



