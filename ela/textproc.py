import string
import numpy as np
import pandas as pd
import scipy.stats.mstats as mstats
import re

from collections import Counter

import nltk
from nltk.corpus import stopwords


def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


LITHO_DESC_COL = 'Lithological Description'

PRIMARY_LITHO_COL = 'Lithology_1'
SECONDARY_LITHO_COL = 'Lithology_2'
PRIMARY_LITHO_NUM_COL = 'Lithology_1_num'
SECONDARY_LITHO_NUM_COL = 'Lithology_2_num'

DEPTH_FROM_COL = 'Depth From (m)'
DEPTH_TO_COL = 'Depth To (m)'
DEPTH_FROM_AHD_COL = 'Depth From (AHD)'
DEPTH_TO_AHD_COL = 'Depth To (AHD)'

EASTING_COL = 'Easting'
NORTHING_COL = 'Northing'

DISTANCE_COL = 'distance'
GEOMETRY_COL = 'geometry'

DEM_ELEVATION_COL = 'DEM_elevation'

WIN_SITE_ID_COL = 'WIN Site ID'

def find_lithologies(df, lithologies):
    '''
    Look for specific lithologies in logs and rank based on their position in the sentence.
    Results are stored in a numpy array "sentence_positions" (rows=records, columns=lithologies)
    '''
    number_of_lithologies=len(lithologies)
    number_of_records=df.shape[0]
    sentence_positions=np.empty([number_of_records,number_of_lithologies])
    lithology_index=0
    for lithology in lithologies:    
        record_index=0
        for index, row in df.iterrows():
            sentence = row[LITHO_DESC_COL]
            words = sentence.split(' ')
            if lithology in words:
                pos = words.index(lithology)
            else:
                pos=-1
            sentence_positions[record_index,lithology_index]=pos
            record_index += 1
        lithology_index +=1
    sentence_positions[sentence_positions == -1] = 'nan'
    return sentence_positions

# sentence_positions
# sentence_positions[sentence_positions == -1] = 'nan'
# sentence_positions[11]
# 0 in sentence_positions[11]
# all(np.isnan(i) for i in sentence_positions[11])
# ranks = mstats.rankdata(np.ma.masked_invalid(sentence_positions[11]))
# print(ranks)
# ranks[ranks == 0] = np.nan
# ranks -= 1
# ranks
# lithologies[np.where(ranks==1)[0][0]]
# any(ranks==1)


def find_lithologies_ranks(df, lithologies, sentence_positions):
    '''
     For each record, identify lithologies ranked in positions 1 & 2 and store in a new dataframe column
    '''
    lithologies_num=np.arange(0,len(lithologies),1)
    record_index=0
    for index, row in df.iterrows():    
        if all(np.isnan(i) for i in sentence_positions[record_index]):
            df.at[record_index,PRIMARY_LITHO_COL] = ''
            df.at[record_index,SECONDARY_LITHO_COL] = ''
            df.at[record_index,PRIMARY_LITHO_NUM_COL] = np.nan
            df.at[record_index,SECONDARY_LITHO_NUM_COL] = np.nan
        else:
            ranks = mstats.rankdata(np.ma.masked_invalid(sentence_positions[record_index]))
            ranks[ranks == 0] = np.nan
            ranks -= 1
            df.at[record_index,PRIMARY_LITHO_COL] = lithologies[np.where(ranks==0)[0][0]]
            df.at[record_index,PRIMARY_LITHO_NUM_COL] = lithologies_num[np.where(ranks==0)[0][0]]
            if any(ranks==1):
                df.at[record_index,SECONDARY_LITHO_COL] = lithologies[np.where(ranks==1)[0][0]]
                df.at[record_index,SECONDARY_LITHO_NUM_COL] = lithologies_num[np.where(ranks==1)[0][0]]
            else:
                df.at[record_index,SECONDARY_LITHO_COL] = ''
                df.at[record_index,SECONDARY_LITHO_NUM_COL] = np.nan
        record_index += 1



def v_find_primary_lithology(v_tokens, lithologies_dict):
    return [find_primary_lithology(x, lithologies_dict) for x in v_tokens]

def v_find_secondary_lithology(v_tokens, prim_litho, lithologies_adjective_dict, lithologies_dict):
    if len(v_tokens) != len(prim_litho):
        raise Error('marker lithology tokens and their primary lithologies must be of same length')
    tokens_and_primary = [(v_tokens[i], prim_litho[i]) for i in range(len(prim_litho))]
    return [find_secondary_lithology(x, lithologies_adjective_dict, lithologies_dict) for x in tokens_and_primary]


def v_word_tokenize(array): 
    res = []
    for y in array:
        res.append(nltk.word_tokenize(y))
    return res

# v_lower = np.vectorize(str.lower)
# Given Python 2.7 we must use:
v_lower = np.vectorize(unicode.lower)

def token_freq(tokens, n_most_common = 50):
    list_most_common=Counter(tokens).most_common(n_most_common)
    return pd.DataFrame(list_most_common, columns=["token","frequency"])

def plot_freq(dataframe, y_log = False, x='token', figsize=(15,10), fontsize=14):
    p = dataframe.plot.bar(x=x, figsize=figsize, fontsize=fontsize)
    if y_log:
        p.set_yscale("log", nonposy='clip')
    return p

def find_word_from_root(tokens, root):
    regex = re.compile('[a-z]*'+root+'[a-z]*')
    xx = list(filter(regex.search, tokens))
    return xx

def plot_freq_for_root(tokens, root, y_log=True):
    sand_terms = find_word_from_root(tokens, root)
    xf = token_freq(sand_terms, len(sand_terms))
    return plot_freq(xf, y_log=y_log)

def split_composite_term(x, joint_re):
    return re.sub("([a-z]+)(" + joint_re + ")([a-z]+)", r"\1 \2 \3", x, flags=re.DOTALL)

def split_with_term(x):
    return split_composite_term(x, 'with')

def v_split_with_term(xlist):
    return [split_with_term(x) for x in xlist]

def clean_lithology_descriptions(description_series, lex):
    expanded_descs = description_series.apply(lex.expand_abbreviations)
    y = expanded_descs.as_matrix()    
    y = v_lower(y)
    y = v_split_with_term(y)
    return y

def find_litho_markers(tokens, regex):
    return list(filter(regex.search, tokens))

def v_find_litho_markers(v_tokens, regex):
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


