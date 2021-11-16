#%%
import pandas as pd
import nltk
import arabic_reshaper
import matplotlib.pyplot as plt 
import re
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from gensim.models import Phrases
from nltk.stem.isri import ISRIStemmer
from bidi.algorithm import get_display
from wordcloud import WordCloud
import seaborn as sns
from tqdm import tqdm
tqdm.pandas()
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# %%

def print_word_cloud_ar(artext_list, word_frequency):
    """Takes a list of Arabic words to print cloud."""
    full_string = ' '.join(artext_list)
    reshaped_text = arabic_reshaper.reshape(full_string)
    artext = get_display(reshaped_text)
    
    # Build the Arabic word cloud
    wordc = WordCloud(font_path='tahoma',background_color='white',width=2000,height=2000).generate(artext)
    wordc.generate_from_frequencies(word_frequency)    
    
    # Draw the word cloud
    plt.imshow(wordc)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    
    plt.show()   
#%%
def print_similar_word_cloud(one_word, topn, model):
    """Takes an Arabic word and print similar word cloud for top number of words {$topn}."""
    temp_tuple=model.wv.most_similar(positive=[one_word], negative=[], topn=topn)
    similar_words=[i[0] for i in temp_tuple]
    word_frequency = {}
    for word_tuple in temp_tuple:
        reshaped_word = arabic_reshaper.reshape(word_tuple[0])
        key = get_display(reshaped_word)
        word_frequency[key] = word_tuple[1]
    print(temp_tuple)
    
    print_word_cloud_ar(similar_words, word_frequency)
#%%

nltk.download('stopwords')
# Extract Arabic stop words
arb_stopwords = set(nltk.corpus.stopwords.words("arabic"))
# Initialize Arabic stemmer
st = ISRIStemmer()
# Load Quran from csv into a dataframe
quran = pd.read_csv('ao.csv', sep='|', header='infer')

# Remove harakat from the verses to simplify the corpus
quran['verse_W_O_H'] = quran['verse'].map(lambda x: re.sub('[ًٌٍَُِّۙ~ْۖۗ]', '', x))               
# Tokinize words from verses and vectorize them
quran['Toknverse'] = quran['verse_W_O_H'].str.split()
# Remove Arabic stop words
quran['fverse'] = quran['Toknverse'].map(lambda x: [w for w in x if w not in arb_stopwords])
quran['cleanT'] = quran['fverse'].apply(lambda i : ' '.join(i))
# Exclude these words from the stemmer
stem_not = ['الله', 'لله', 'إلهكم', 'اله', 'لله', 'إلهكم', 'إله', 'بالله', 'ولله']
# You can filter for one surah too if you want!
verses = quran['verse'].values.tolist()



#%%

model = Word2Vec(quran['fverse'], min_count=5, window=10, workers=16, alpha=0.1, sg=1, hs=1)
#%%
wordo = "موسى"


print_similar_word_cloud(wordo, 15,model)

# %%

# %%
