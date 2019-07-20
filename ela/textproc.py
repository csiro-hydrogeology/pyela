import string
import sys
import numpy as np
import pandas as pd
import scipy.stats.mstats as mstats
import re

from collections import Counter

import nltk
from nltk.corpus import stopwords


def replace_punctuations(text, replacement=' '):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, replacement)
    return text

def remove_punctuations(text):
    return replace_punctuations(text, '')

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

def v_remove_punctuations(textlist):
    return [remove_punctuations(x) for x in textlist]

def v_replace_punctuations(textlist, replacement=' '):
    return [replace_punctuations(x, replacement) for x in textlist]

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


def flat_list_tokens(descriptions):
    """Convert a collection of strings to a flat list of tokens. English NLTK stopwords.

    Args:
        descriptions (iterable of str): lithology descriptions.

    Returns:
        list: List of tokens.

    """
    vt = v_word_tokenize(descriptions)
    flat = np.concatenate(vt)
    stoplist = stopwords.words('english')
    exclude = stoplist + ['.',',',';',':','(',')','-']
    flat = [word for word in flat if word not in exclude]
    return flat


def match_and_sample_df(df, litho_class_name, colname=PRIMARY_LITHO_COL, out_colname=None, size=50, seed=None):
    """Sample a random subset of rows where the lithology column matches a particular class name.

        Args:
            df (pandas data frame): bore lithology data  with columns named PRIMARY_LITHO_COL
    
        Returns:
            a list of strings, compound primary+optional_secondary lithology descriptions e.g. 'sand/clay', 'loam/'
    """
    df_test = df.loc[ df[colname] == litho_class_name ]
    y = df_test.sample(n=size, frac=None, replace=False, weights=None, random_state=seed)
    if not out_colname is None:
        y = y[LITHO_DESC_COL]
    return y


def find_regex_df(df, expression, colname):
    """Sample a random subset of rows where the lithology column matches a particular class name.

        Args:
            df (pandas data frame): bore lithology data  with columns named PRIMARY_LITHO_COL
    
        Returns:
            dataframe:
    """
    tested = df[colname].values
    regex = re.compile(expression)
    xx = [(regex.match(x) is not None) for x in tested]
    df_test = df.loc[xx]
    return df_test
