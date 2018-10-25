import os
import pandas as pd
import numpy as np
import sys
from datetime import datetime
from striplog import Lexicon

pkg_dir = os.path.join(os.path.dirname(__file__),'..')

sys.path.append(pkg_dir)

from ela.textproc import *

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

