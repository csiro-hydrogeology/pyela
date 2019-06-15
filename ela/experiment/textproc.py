import numpy as np
import string
import pandas as pd
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
from wordcloud import WordCloud,STOPWORDS


import nltk
from nltk.corpus import stopwords



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
        
        return output
            
        
        
        
        
        

