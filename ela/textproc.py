import string
import sys
import numpy as np
import pandas as pd
import scipy.stats.mstats as mstats
import re
import gensim
import tensorflow as tf
from keras import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import utils
from keras import regularizers
from keras import optimizers
from wordcloud import WordCloud,STOPWORD

from collections import Counter

import nltk
from nltk.corpus import stopwords


def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


LITHO_DESC_COL = u'Lithological Description'

PRIMARY_LITHO_COL = u'Lithology_1'
SECONDARY_LITHO_COL = u'Lithology_2'
PRIMARY_LITHO_NUM_COL = u'Lithology_1_num'
SECONDARY_LITHO_NUM_COL = u'Lithology_2_num'

DEPTH_FROM_COL = u'Depth From (m)'
DEPTH_TO_COL = u'Depth To (m)'
DEPTH_FROM_AHD_COL = u'Depth From (AHD)'
DEPTH_TO_AHD_COL = u'Depth To (AHD)'

EASTING_COL = u'Easting'
NORTHING_COL = u'Northing'

DISTANCE_COL = u'distance'
GEOMETRY_COL = u'geometry'

DEM_ELEVATION_COL = u'DEM_elevation'

WIN_SITE_ID_COL = u'WIN Site ID'



def v_find_primary_lithology(v_tokens, lithologies_dict):
    """Vectorised function to find a primary lithology in a list of tokenised sentences.

    Args:
        v_tokens (iterable of iterable of str): the list of tokenised sentences.
        lithologies_dict (dict): dictionary, where keys are exact markers as match for lithologies. Values are the lithology classes. 

    Returns:
        list: list of primary lithologies if dectected. empty string for none.

    """
    return [find_primary_lithology(x, lithologies_dict) for x in v_tokens]

def v_find_secondary_lithology(v_tokens, prim_litho, lithologies_adjective_dict, lithologies_dict):
    """Vectorised function to find a secondary lithology in a list of tokenised sentences.

    Args:
        v_tokens (iterable of iterable of str): the list of tokenised sentences.
        prim_litho (list of str): the list of primary lithologies already detected for v_tokens. The secondary lithology cannot be the same as the primary.
        lithologies_adjective_dict (dict): dictionary, where keys are exact, "clear" markers for secondary lithologies (e.g. 'clayey'). Keys are the lithology classes. 
        lithologies_dict (dict): dictionary, where keys are exact markers as match for lithologies. Values are the lithology classes.

    Returns:
        list: list of secondary lithologies if dectected. empty string for none.

    """
    if len(v_tokens) != len(prim_litho):
        raise Error('marker lithology tokens and their primary lithologies must be of same length')
    tokens_and_primary = [(v_tokens[i], prim_litho[i]) for i in range(len(prim_litho))]
    return [find_secondary_lithology(x, lithologies_adjective_dict, lithologies_dict) for x in tokens_and_primary]


def v_word_tokenize(descriptions): 
    """Vectorised tokenisation of lithology descriptions.

    Args:
        descriptions (iterable of str): lithology descriptions.

    Returns:
        list: list of lists of tokens in the NLTK.

    """
    return [nltk.word_tokenize(y) for y in descriptions]

v_lower = None
"""vectorised, unicode version to lower case strings
"""

if(sys.version_info.major > 2):
    v_lower = np.vectorize(str.lower)
    """vectorised, unicode version to lower case strings
    """
else:
    # Given Python 2.7 we must use:
    v_lower = np.vectorize(unicode.lower)
    """vectorised, unicode version to lower case strings
    """

def token_freq(tokens, n_most_common = 50):
    """Gets the most frequent (counts) tokens 

    Args:
        tokens (iterable of str): the list of tokens to analyse for frequence.
        n_most_common (int): subset to the this number of most frequend tokens

    Returns:
        pandas DataFrame: columns=["token","frequency"]

    """
    list_most_common=Counter(tokens).most_common(n_most_common)
    return pd.DataFrame(list_most_common, columns=["token","frequency"])

def plot_freq(dataframe, y_log = False, x='token', figsize=(15,10), fontsize=14):
    p = dataframe.plot.bar(x=x, figsize=figsize, fontsize=fontsize)
    if y_log:
        p.set_yscale("log", nonposy='clip')
    return p

def find_word_from_root(tokens, root):
    """Filter token (words) to retain only those containing a root term 

    Args:
        tokens (iterable of str): the list of tokens.
        root (str): regular expression for the root term, to look for (e.g 'clay' or 'cl(a|e)y'), which will be padded with '[a-z]*' for searching

    Returns:
        a list: terms matching the root term.

    """
    regex = re.compile('[a-z]*'+root+'[a-z]*')
    xx = list(filter(regex.search, tokens))
    return xx

def plot_freq_for_root(tokens, root, y_log=True):
    sand_terms = find_word_from_root(tokens, root)
    xf = token_freq(sand_terms, len(sand_terms))
    return plot_freq(xf, y_log=y_log)

def split_composite_term(x, joint_re = 'with'):
    """Break terms that are composites padding several words without space. This has been observed in one case study but may not be prevalent.

    Args:
        x (str): the term to split if matching, e.g. 'claywithsand' to 'clay with sand'
        joint_re (str): regular expression for the word used as fusing join, typically 'with' 

    Returns:
        split wording (str): tokens split from the joining term.

    """
    return re.sub("([a-z]+)(" + joint_re + ")([a-z]+)", r"\1 \2 \3", x, flags=re.DOTALL)

def split_with_term(x):
    return split_composite_term(x, 'with')

def v_split_with_term(xlist):
    return [split_with_term(x) for x in xlist]

def clean_lithology_descriptions(description_series, lex):
    """Preparatory cleanup of lithology descriptions for further analysis: 
    replace abbreviations and misspelling according to a lexicon, 
    and transform to lower case

    Args:
        description_series (iterable of str, or pd.Series): lithology descriptions 
        lex (striplog.Lexicon): an instance of striplog's Lexicon

    Returns:
        (iterable of str): processed descriptions.
    """
    if isinstance(description_series, list):
        y = [lex.expand_abbreviations(x) for x in description_series]
    else:
        expanded_descs = description_series.apply(lex.expand_abbreviations)
        y = expanded_descs.values
    y = v_lower(y)
    y = v_split_with_term(y)
    return y

def find_litho_markers(tokens, regex):
    return list(filter(regex.search, tokens))

def v_find_litho_markers(v_tokens, regex):
    """Find lithology lithology terms that match a regular expression

    Args:
        v_tokens (iterable of iterable of str): the list of tokenised sentences.
        regex (regex): compiles regular expression  e.g. re.compile('sand|clay')

    Returns:
        (iterable of iterable of str): tokens found to be matching the expression
    """
    return [find_litho_markers(t,regex) for t in v_tokens]


# I leave 'basalt' out, as it was mentioned it may be a mistake in the raw log data.
DEFAULT_LITHOLOGIES = ['sand','sandstone','clay','limestone','shale','coffee','silt','gravel','granite','soil','loam']

DEFAULT_ANY_LITHO_MARKERS_RE = r'sand|clay|ston|shale|basalt|coffee|silt|granit|soil|gravel|loam|mud|calca|calci'

DEFAULT_LITHOLOGIES_DICT = dict([(x,x) for x in DEFAULT_LITHOLOGIES])
DEFAULT_LITHOLOGIES_DICT['sands'] = 'sand'
DEFAULT_LITHOLOGIES_DICT['clays'] = 'clay'
DEFAULT_LITHOLOGIES_DICT['shales'] = 'shale'
DEFAULT_LITHOLOGIES_DICT['claystone'] = 'clay'
DEFAULT_LITHOLOGIES_DICT['siltstone'] = 'silt'
DEFAULT_LITHOLOGIES_DICT['limesand'] = 'sand' # ??
DEFAULT_LITHOLOGIES_DICT['calcarenite'] = 'limestone' # ??
DEFAULT_LITHOLOGIES_DICT['calcitareous'] = 'limestone' # ??
DEFAULT_LITHOLOGIES_DICT['mudstone'] = 'silt' # ??
DEFAULT_LITHOLOGIES_DICT['capstone'] = 'limestone' # ??
DEFAULT_LITHOLOGIES_DICT['ironstone'] = 'sandstone' # ??
DEFAULT_LITHOLOGIES_DICT['topsoil'] = 'soil' # ??

def find_primary_lithology(tokens, lithologies_dict):
    """Find a primary lithology in a tokenised sentence.

    Args:
        v_tokens (iterable of iterable of str): the list of tokenised sentences.
        lithologies_dict (dict): dictionary, where keys are exact markers as match for lithologies. Keys are the lithology classes. 

    Returns:
        list: list of primary lithologies if dectected. empty string for none.

    """
    keys = lithologies_dict.keys()
    for x in tokens:
        if x in keys:
            return lithologies_dict[x]
    return ''


DEFAULT_LITHOLOGIES_ADJECTIVE_DICT = {
    'sandy' :  'sand',
    'clayey' :  'clay',
    'clayish' :  'clay',
    'shaley' :  'shale',
    'silty' :  'silt',
    'gravelly' :  'gravel'
}

def find_secondary_lithology(tokens_and_primary, lithologies_adjective_dict, lithologies_dict):
    """Find a secondary lithology in a tokenised sentence.

    Args:
        tokens_and_primary (tuple ([str],str): tokens and the primary lithology
        lithologies_adjective_dict (dict): dictionary, where keys are exact, "clear" markers for secondary lithologies (e.g. 'clayey'). Keys are the lithology classes. 
        lithologies_dict (dict): dictionary, where keys are exact markers as match for lithologies. Keys are the lithology classes.

    Returns:
        str: secondary lithology if dectected. empty string for none.

    """
    tokens, prim_litho = tokens_and_primary
    if prim_litho == '': # cannot have a secondary lithology if no primary
        return ''
    # first, let's look at adjectives, more likely to semantically mean a secondary lithology
    keys = lithologies_adjective_dict.keys()
    for x in tokens:
        if x in keys:
            litho_class = lithologies_adjective_dict[x]
            if litho_class != prim_litho:
                return litho_class
    # then, as a fallback let's look at a looser set of terms to find a secondary lithology
    keys = lithologies_dict.keys()
    for x in tokens:
        if x in keys:
            litho_class = lithologies_dict[x]
            if litho_class != prim_litho:
                return litho_class
    return ''


class Model:
    def __init__(self,train_data,maxlen):
        self.train_data=train_data
        self.maxlen=maxlen
        self.embedding_model=None
        self.ml_model=None
        
    def load_data(self):
        """
        Loading the data into a dataframe

        Input
        path: path to the test data(String)

        Output
        train_data: return a pandas Dataframe
        """
        print(self.train_data.head())
    
    #referenced from https://stackoverflow.com/questions/16645799/how-to-create-a-word-cloud-from-a-corpus-in-python
    def show_wordcloud(self, title = None):
        """
        depicting wordclouds of the input data

        Input
        data: input pandas Dataframe
        """
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(
            background_color='white',
            stopwords=stopwords,
            max_words=200,
            max_font_size=40, 
            scale=3,
            random_state=1 # chosen at random by flipping a coin; it was heads
        ).generate(str(self.train_data))

        fig = plt.figure(1, figsize=(12, 12))
        plt.axis('off')
        if title: 
            fig.suptitle(title, fontsize=20)
            fig.subplots_adjust(top=2.3)

        plt.imshow(wordcloud)
        plt.show()

        
    def preprocessor(self,data):
        """
        Tokenizing the sentences using regular expressions and NLTK library

        Input
        text: list of descriptions

        Output:
        alphabet_tokens: list of tokens
        """
        __tokenization_pattern = r'''(?x)          # set flag to allow verbose regexps
            \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
          | (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
          | \w+(?:-\w+)*        # words with optional internal hyphens
          | \.\.\.              # ellipsis
          | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
        '''

        ## call it using tokenizer.tokenize
        tokenizer = nltk.tokenize.regexp.RegexpTokenizer(__tokenization_pattern)
        tokens = tokenizer.tokenize(data)
        tokens=[token.lower() for token in tokens if token.isalpha()]
        alphabet_tokens = [token for token in tokens if token.isalpha()]
        #en_stopwords = set(nltk.corpus.stopwords.words('english'))
        #non_stopwords = [word for word in alphabet_tokens if not word in en_stopwords]
        #stemmer = nltk.stem.snowball.SnowballStemmer("english")
        #stems = [str(stemmer.stem(word)) for word in non_stopwords]
        if len(alphabet_tokens)==1:
            return alphabet_tokens[0]
        else:
            return alphabet_tokens
    
    
    def transform_data(self):
        """
        Factorizing the simplified lithologies into numerical equivalents

        Input
        data: input pandas dataframe

        Output
        tuple containing the transformed data
        """
        self.train_data['Lithology_original']=self.train_data['Lithology_original'].replace(np.nan,'',regex=True)
        self.train_data['Lithology_original'] =self.train_data['Lithology_original'].apply(self.preprocessor)
        self.train_data['Simplified_lithology']=self.train_data['Simplified_lithology'].replace(np.nan,'Unknown',regex=True)
        self.train_data['Simplified_lithology']=self.train_data['Simplified_lithology'].apply(self.preprocessor).astype(str)
        self.train_data['Simplified_lithology'],self.label=pd.factorize(self.train_data['Simplified_lithology'])
        self.list_of_descriptions=self.train_data['Lithology_original'].tolist()
        self.list_of_simple_lithology=self.train_data['Simplified_lithology'].tolist()
    
    
    def generate_embeddings(self):
        """
        Generating FastText(vectorized version of each word) model from the vocabulary in the data

        Input
        list_of_descriptions: transformed descriptions
        list_of_simple_lithology: transformed simple lithologies

        Output
        model: Gensim fasttext model

        """
        data=[]
        for x in self.list_of_descriptions:
            temp=[]
            if(isinstance(x,list)):
                for y in x:
                    temp.append(y.lower())
                data.append(temp)
        for x in self.list_of_simple_lithology:
            temp=[]
            if(isinstance(x,list)):
                for y in x:
                    temp.append(y.lower())
                data.append(temp)
            if(isinstance(x,float)):
                print(x)
        self.embedding_model=gensim.models.FastText(data,min_count=1,size=100,window=3)

    
    
    def split_data(self):
        """
        Splitting the data into train and test

        Input
        train_data: Pandas dataframe

        Output
        tuple containing train and test data 
        """
        msk = np.random.rand(len(self.train_data)) < 0.75
        self.train_X = self.train_data.Lithology_original[msk]
        self.test_X = self.train_data.Lithology_original[~msk]
        y=self.train_data['Simplified_lithology']
        self.train_y = y[msk]
        self.test_y=y[~msk]

        

    
    def tokenize_input_data(self):
        """
        Indexing each token in the descriptions

        Input
        train_X: list of input descriptions
        test_X : list of input descriptions

        Output
        Tuple containing indexed versions of the inputs
        """
        self.tokenizer=Tokenizer(num_words=3000)    
        self.tokenizer.fit_on_texts(self.train_X)
        self.train_X=self.tokenizer.texts_to_sequences(self.train_X)
        self.test_X=self.tokenizer.texts_to_sequences(self.test_X)
    
    
    def label_to_id(self):
        """
        Indexing each label in the target(simplified lithology)

        Input
        train_y: list of labels
        test_y: list of labels

        Output
        tuple containing indexed versions of the input
        """
        
        self.train_y=utils.to_categorical(self.train_y.tolist(),11,dtype='int')
        self.test_y=utils.to_categorical(self.test_y.tolist(),11,dtype='int')
    
    
    def pad_sentences(self):
        """
        Adding padding to the descriptions so that each description is of the same length(maxlen)

        Input
        train_X: list of descriptions
        test_X: list of descriptions
        maxlen: int (maximum length of the descriptions)

        Output
        Tuple containing transformed versions of the input
        """
        self.train_X= pad_sequences(self.train_X, padding='post', maxlen=self.maxlen)
        self.test_X= pad_sequences(self.test_X, padding='post', maxlen=self.maxlen)

    
    
    def create_embedding_matrix(self):
        """
        Creating an embedding matrix to be fed into the neural network

        Input
        model: gensim word2vec model

        embedding_matrix: matrix depicting the embeddings
        """
        self.embedding_matrix=np.zeros((len(self.embedding_model.wv.vocab),100))
        for x,y in self.embedding_model.wv.vocab.items():
            if x in self.tokenizer.word_counts.keys():
                self.embedding_matrix[self.tokenizer.word_index[x]]=np.array(self.embedding_model.wv[x], dtype=np.float32)[:100]

        
    
    
    def define_learning_model(self):
        """
        Describing the deep learning model using Keras

        Input
        model:gensim word2vec model
        embedding_matrix: matrix of embeddings
        maxlen: maximum length of sentences

        Output
        lstm_model: deep learning model
        """
        self.ml_model=Sequential()
        self.ml_model.add(layers.Embedding(len(self.embedding_model.wv.vocab), 100, 
                                   weights=[self.embedding_matrix],
                                   input_length=self.maxlen,
                                   trainable=False))
        self.ml_model.add(layers.LSTM(100))
        #model.add(layers.Dropout(0.3))
        #model.add(layers.LSTM(100,activation='tanh',recurrent_activation='sigmoid'))
        self.ml_model.add(layers.Dropout(0.3))

        #model.add(layers.GlobalAveragePooling1D())
        self.ml_model.add(layers.Dense(11,activation='softmax'))
        #self.ml_model.add(layers.Softmax())
        #model.add(layers.Flatten())
        adam=optimizers.Adam(lr=0.001)
        self.ml_model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.ml_model.summary()
    
    
    def calculate_accuracy(self):
        """
        Calculating the accuracy of the model.

        Input
        train_X: list of descriptions
        train_y: list of labels

        Output:
        history: model after fitting the data

        """
        msk = np.random.rand(len(self.train_X)) < 0.75
        validation_data_X=self.train_X[~msk]
        validation_data_Y=self.train_y[~msk]
        self.history = self.ml_model.fit(self.train_X[msk],self.train_y[msk],
                            epochs=10,
                            verbose=2,
                           validation_data=(validation_data_X,validation_data_Y))
        loss, accuracy = self.ml_model.evaluate(self.train_X, self.train_y, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = self.ml_model.evaluate(self.test_X, self.test_y, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))


    
    
    #used as reference from https://www.tensorflow.org/tutorials/keras/basic_text_classification
    def plot_loss(self):
        """
        Plot the training and validation loss w.r.t epochs

        Input
        model: deep learning model
        """
        history_dict = self.history.history
        history_dict.keys()
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(loss) + 1)

        # "bo" is for "blue dot"
        plt.plot(epochs, loss, 'bo', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
        
        
    
    #used as reference from https://www.tensorflow.org/tutorials/keras/basic_text_classification
    def plot_accuracy(self):
        """
        Plot the training and validation accuracy w.r.t epochs

        Input
        model: deep learning model
        """
        plt.clf()   # clear figure
        history_dict = self.history.history
        history_dict.keys()
        acc = history_dict['acc']
        val_acc = history_dict['val_acc']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()
        
        
    def initialise_model(self):
        """
        
        Develop the model based on the input data
        
        """
        self.load_data()
        self.transform_data()
        self.generate_embeddings()
        self.split_data()
        self.tokenize_input_data()
        self.label_to_id()
        self.pad_sentences()
        self.create_embedding_matrix()
        self.define_learning_model()
        self.calculate_accuracy()
        
        
        
        
        
        
        
        
    def predict(self,data):
        """
        Predict simplified lithologies for input data
        
        """
        
        data['Description']=data['Description'].replace(np.nan,'',regex=True)
        data['Description']=data['Description'].astype(str)
        predict_X=self.tokenizer.texts_to_sequences(data['Description'])
        
        predict_X=pad_sequences(predict_X,padding='post',maxlen=self.maxlen)
        output=self.ml_model.predict_classes(predict_X)
        simplified_lithology=[]
        for x in output:
            simplified_lithology.append(self.label[x])
        data['Simplified_Lithology']=pd.Series(simplified_lithology)
        data.to_csv('prediction_file.csv',index=False)
        
        

    def predict_certainity(self,data):
        
        data['Description']=data['Description'].replace(np.nan,'',regex=True)
        data['Description']=data['Description'].astype(str)
        predict_X=self.tokenizer.texts_to_sequences(data['Description'])
        
        predict_X=pad_sequences(predict_X,padding='post',maxlen=self.maxlen)
        output=self.ml_model.predict_proba(predict_X)
        
        print(output)
            
        
        
        
        
        

