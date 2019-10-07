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
import utils_2 as utils


#%%
test_sentences = [
    'the old man spoke to me',
    'me to spoke man old the',
    'old man me old man me',
]


#%%
def log_prob_of_sentence(sentence, bigram_log_dict):
    # get the sentence bigrams
    s_tokens, s_bigrams = utils.sentence_to_bigrams(sentence)

    # add the log probabilites of the bigrams in the sentence
    total_log_prob = 0.
    for bg in s_bigrams:
        if bg in bigram_log_dict:
            total_log_prob = total_log_prob + bigram_log_dict[bg]
        else:
            total_log_prob = total_log_prob + bigram_log_dict['<unk>']
    return total_log_prob


#%%



