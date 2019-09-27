"""Text processing features for lithology analysis.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension
"""

import string
import sys
import numpy as np
import pandas as pd
import scipy.stats.mstats as mstats
import re

from collections import Counter

import striplog
import nltk
from nltk.corpus import stopwords


def replace_punctuations(text, replacement=' '):
    """Replace the punctuations (``string.punctuation``) in a string."""
    for punctuation in string.punctuation:
        text = text.replace(punctuation, replacement)
    return text

def remove_punctuations(text):
    """Remove the punctuations (``string.punctuation``) in a string."""
    return replace_punctuations(text, '')

LITHO_DESC_COL = u'Lithological Description'
"""Default column name expected in lithodescription data frames"""

PRIMARY_LITHO_COL = u'Lithology_1'
"""Default column name expected in lithodescription data frames"""
SECONDARY_LITHO_COL = u'Lithology_2'
"""Default column name expected in lithodescription data frames"""
PRIMARY_LITHO_NUM_COL = u'Lithology_1_num'
"""Default column name expected in lithodescription data frames"""
SECONDARY_LITHO_NUM_COL = u'Lithology_2_num'
"""Default column name expected in lithodescription data frames"""

DEPTH_FROM_COL = u'Depth From (m)'
"""Default column name expected in lithodescription data frames"""
DEPTH_TO_COL = u'Depth To (m)'
"""Default column name expected in lithodescription data frames"""
DEPTH_FROM_AHD_COL = u'Depth From (AHD)'
"""Default column name expected in lithodescription data frames"""
DEPTH_TO_AHD_COL = u'Depth To (AHD)'
"""Default column name expected in lithodescription data frames"""

EASTING_COL = u'Easting'
"""Default column name expected in lithodescription data frames"""
NORTHING_COL = u'Northing'
"""Default column name expected in lithodescription data frames"""

DISTANCE_COL = u'distance'
"""Default column name expected in lithodescription data frames"""
GEOMETRY_COL = u'geometry'
"""Default column name expected in lithodescription data frames"""

DEM_ELEVATION_COL = u'DEM_elevation'
"""Default column name expected in lithodescription data frames"""

# columns in the BoM NGIS data model
# http://www.bom.gov.au/water/regulations/dataDelivery/document/NgisDiagramv2.3.pdf

HYDRO_CODE_COL = u'HydroCode'
"""Jurisdictional bore and pipe identifier within NGIS geodatabase"""

HYDRO_ID_COL = u'HydroID'
"""Unique feature identifier within NGIS geodatabase"""

BORE_ID_COL = u'BoreID'
"""Numeric identifier in lithology logs corresponding to the HydroID of NGIS_Bore feature"""

# WIN_SITE_ID_COL = u'WIN Site ID'

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
    """Plot a sorted histogram of work frequencies

    Args:
        dataframe (pandas dataframe): frequency of tokens, typically with colnames ["token","frequency"]
        y_log (bool): should there be a log scale on the y axis
        x (str): name of the columns with the tokens (i.e. words)
        figsize (tuple):
        fontsize (int):

    Returns:
        barplot: plot

    """
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
    """Plot a sorted histogram of work frequencies

    Args:
        tokens (iterable of str): the list of tokens.
        root (str): regular expression for the root term, to look for (e.g 'clay' or 'cl(a|e)y'), which will be padded with '[a-z]*' for searching
        y_log (bool): should there be a log scale on the y axis

    Returns:
        barplot: plot

    """
    matching_terns = find_word_from_root(tokens, root)
    xf = token_freq(matching_terns, len(matching_terns))
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
    """split words that are joined by a with, i.e. 'sandwithclay'
    Args:
        x (str): the term to split if matching, e.g. 'claywithsand' to 'clay with sand'

    Returns:
        split wording (str): tokens split from the joining term.
    """
    return split_composite_term(x, 'with')

def v_split_with_term(xlist):
    """split words that are joined by a with, i.e. 'sandwithclay'
    Args:
        xlist (iterable of str): the terms to split if matching, e.g. 'claywithsand' to 'clay with sand'

    Returns:
        split tokens (list of str): tokens split from the joining term.
    """
    return [split_with_term(x) for x in xlist]

def v_remove_punctuations(textlist):
    """vectorised function to remove punctuations
    Args:
        textlist (iterable of str): list of terms

    Returns:
        (list):
    """
    return [remove_punctuations(x) for x in textlist]

def v_replace_punctuations(textlist, replacement=' '):
    """vectorised function to replace punctuations
    Args:
        textlist (iterable of str): list of terms

    Returns:
        (list):
    """
    return [replace_punctuations(x, replacement) for x in textlist]

def clean_lithology_descriptions(description_series, lex = None):
    """Preparatory cleanup of lithology descriptions for further analysis

    Replace abbreviations and misspelling according to a lexicon,
    and transform to lower case

    Args:
        description_series (iterable of str, or pd.Series): lithology descriptions
        lex (striplog.Lexicon): an instance of striplog's Lexicon

    Returns:
        (iterable of str): processed descriptions.
    """
    if lex is None:
        lex = striplog.Lexicon.default()
    if isinstance(description_series, list):
        y = [lex.expand_abbreviations(x) for x in description_series]
    else:
        expanded_descs = description_series.apply(lex.expand_abbreviations)
        y = expanded_descs.values
    y = v_lower(y)
    y = v_split_with_term(y)
    return y

def find_litho_markers(tokens, regex):
    """Find lithology lithology terms that match a regular expression

    Args:
        tokens (iterable of str): the list of tokenised sentences.
        regex (regex): compiles regular expression  e.g. re.compile('sand|clay')

    Returns:
        (list of str): tokens found to be matching the expression
    """
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

def as_numeric(x):
    if isinstance(x, float):
        return x
    if x == 'None':
        return np.nan
    elif x is None:
        return np.nan
    elif isinstance(x, str):
        return float(x)
    else:
        return float(x)

def columns_as_numeric(df, colnames=None):
    """Process some columns to make sure they are numeric. In-place changes.

        Args:
            df (pandas data frame): bore lithology data
            colnames (iterable of str): column names
    """
    colnames = colnames or [DEPTH_FROM_COL, DEPTH_TO_COL]
    for colname in colnames:
        df[colname] = df[colname].apply(as_numeric)

