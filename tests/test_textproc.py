import os
import pandas as pd
import sys
from striplog import Lexicon

pkg_dir = os.path.join(os.path.dirname(__file__),'..')

sys.path.insert(0, pkg_dir)

from ela.textproc import *

# To avoid failing test on Travis:

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# try:
#     nltk.data.find('tokenizers/averaged_perceptron_tagger')
# except LookupError:
#     nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('tokenizers/stopwords')
except LookupError:
    nltk.download('stopwords')

def test_litho_marker_detection():
    prim_classes = {
        'sand': 0,
        'clay': 1,
        'sandstone': 2
    }
    adjective_classes = {
        'sandy': 0,
        'clayey': 1,
        'choc': 3
    }
    sentences = [
        ['sand','with','greyish','clay'],
        ['clayey','blueish','sandstone','with','coarse','choc','chips']
    ]
    prims= v_find_primary_lithology(sentences, prim_classes)
    assert prims[0] == 0
    assert prims[1] == 2
    seconds = v_find_secondary_lithology(sentences, prims, adjective_classes, prim_classes)
    assert seconds[0] == 1
    assert seconds[1] == 1 # clayey, not choc



def test_v_word_tokenize():
    descriptions = ['yellow, slightly clayey sand','75% sand, 20% silt, 5% gravel']
    tkns = v_word_tokenize(descriptions)
    assert len(tkns) == 2
    assert len(tkns[0]) == 5
    assert len(tkns[1]) == 11


def test_find_word_from_root():
    tkns = ['gravel', 'sand', 'clay', 'clayish', 'vclayey', 'vvclay', 'silt']
    terms = find_word_from_root(tkns, 'clay')
    assert set(['clay', 'clayish', 'vclayey', 'vvclay']) == set(terms)


def test_plot_freq_for_root():
    tkns = ['gravel', 'sand', 'clay', 'clayish', 'vclayey', 'vvclay', 'silt']
    p = plot_freq_for_root(tkns, root='clay')
    assert p is not None


def test_split_composite_term():
    x = split_composite_term('claywithsand', 'with')
    assert x == 'clay with sand'

def test_clean_lithology_descriptions():
    lex = Lexicon.default()
    descriptions = [
        'Sand with qtz',
        'Sand calc grey fg mg',
        'Clay very sandy, grey greenish, pyritic',
        'sandwithclay'
    ]
    x = clean_lithology_descriptions(descriptions, lex)
    assert x[0] == 'sand with quartz'
    assert x[1] == 'sand calcitareous grey fg mg'
    assert x[2] == 'clay very sandy, grey greenish, pyritic'
    assert x[3] == 'sand with clay'


def test_v_find_litho_markers():
    v_tokens = [
        ['slightly', 'clayey', 'grey', 'sand', 'with', 'a', 'touch', 'of', 'silt'],
        ['ironstone', 'with', 'boulders']
    ]
    regex = re.compile('sand|clay|silt|iron')
    terms = v_find_litho_markers(v_tokens, regex)
    assert terms[0][0] == 'clayey'
    assert terms[0][1] == 'sand'
    assert terms[0][2] == 'silt'
    assert terms[1][0] == 'ironstone'

def test_flatten_strings():
    descs = [
        'clay, sand',
        'clay with loam  ;',
        'uranium'
    ]
    tk = flat_list_tokens(descs)
    assert set(tk) == set(['clay', 'loam', 'uranium', 'sand'])


def test_replace_punctuations():
    textlist = ['Lots of basalt.driller joe 24/10/1982','clay/loam vfine silt. black-brown']
    rpl = v_replace_punctuations(textlist)
    assert rpl[0] == 'Lots of basalt driller joe 24 10 1982'
    assert rpl[1] == 'clay loam vfine silt  black brown'
    rpl = v_replace_punctuations(textlist, replacement='R')
    assert rpl[0] == 'Lots of basaltRdriller joe 24R10R1982'



