import os
import re
import codecs
import random
from utils import create_dico, create_mapping, zero_digits, create_input
from utils import iob2, iob_iobes, iob_bin
from collections import defaultdict

from ccg_nlpy.core.text_annotation import TextAnnotation

from ccg_nlpy.core.view import View

def load_sentences(path, lower, zeros, ratio=1.0):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """

    file_list = os.listdir(path)
    sentences = []
    label_list = set()
    for doc in file_list:
        print("Reading " + os.path.join(path, doc))
        document = TextAnnotation(json_str=open(os.path.join(path, doc)).read())
        ner_labels = document.view_dictionary['NER_CONLL'].cons_list
        if ner_labels is None:
            ner_labels = []
        ner_dict = {}
        for ner_constituent in ner_labels:
            for span in range(ner_constituent['start'], ner_constituent['end']):
                if span-ner_constituent['start'] == 0:
                    ner_dict[span] = "B-" + ner_constituent['label']
                else:
                    ner_dict[span] = "I-" + ner_constituent['label']
                if ner_dict[span] not in label_list:
                    label_list.add(ner_dict[span])
                    print(ner_dict[span])
        try:
            sentences_cons_list = document.view_dictionary['SENTENCE'].cons_list
        except KeyError as e:
            sentences_cons_list = []
            start = 0
            for end in document.sentence_end_position:
                sent = " ".join(document.tokens[start:end])
                sentences_cons_list.append({'tokens': sent, 'start': start, 'end': end})
                start = end
        for sent_constituent in sentences_cons_list:
            sentence = []
            sent = re.split("\s+", sent_constituent['tokens'])
            start = sent_constituent['start']
            end = sent_constituent['end']
            for token, span in zip(sent, range(start, end)):
                if span in ner_dict:
                    sentence.append([token, ner_dict[span]])
                else:
                    sentence.append([token, "O"])
            sentences.append(sentence)

    random.shuffle(sentences)
    train_sentences = sentences[:int(ratio*len(sentences))]
    dev_sentences = sentences[int(ratio*len(sentences)):]
    return train_sentences, dev_sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        elif tag_scheme == 'bin':
            new_tags = iob_bin(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico)
    print "Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    )
    return dico, word_to_id, id_to_word


def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    char_to_id, id_to_char = create_mapping(dico)
    print "Found %i unique characters" % len(dico)
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print "Found %i unique named entity tags" % len(dico)
    return dico, tag_to_id, id_to_tag


def brown_mapping(filename):
    """
    Create Brown mapping
    """
    with open(filename) as f:
        data = f.readlines()
    data = [(re.split("\s+", line.strip())[1], re.split("\s+", line.strip())[0]) for line in data if len(line.strip()) > 0]
    dict_brown = defaultdict(lambda: 0)
    brown_to_id = {"<UNK>": 0}
    id_to_brown = {0: "<UNK>"}
    idx = 0
    for (entity, tag) in data:
        if tag not in brown_to_id:
            brown_to_id[tag] = idx + 1
            id_to_brown[idx + 1] = tag
            idx += 1
        dict_brown[entity.lower()] = brown_to_id[tag]
    return dict_brown, brown_to_id, id_to_brown        


def gazetteer_mapping(filename):
    """
    Create gazetteer mapping
    """
    with open(filename) as f:
        data = f.readlines()
    data = [(line.strip().split(";")[1], line.strip().split(";")[2]) for line in data if len(line.strip()) > 0]
    dict_gtr = defaultdict(lambda: 0)
    gtr_to_id = {"<UNK>": 0, "G": 1}
    id_to_gtr = {0: "<UNK>", 1: "G"}
    idx = 0
    # for (_, tag) in data:
    #     if "B-" + tag not in gtr_to_id:
    #         gtr_to_id["B-" + tag] = idx + 1
    #         id_to_gtr[idx + 1] = "B-" + tag
    #         idx += 1
    #     if "I-" + tag not in gtr_to_id:
    #         gtr_to_id["I-" + tag] = idx + 1
    #         id_to_gtr[idx + 1] = "I-" + tag
    #         idx += 1

    for (entity, tag) in data:
        token = re.split("\s+", entity)
        for idx in range(len(token)):
            if idx == 0:
                dict_gtr[token[idx].lower()] = 1 # gtr_to_id["B-" + tag]
            else:
                dict_gtr[token[idx].lower()] = 1 # gtr_to_id["I-" + tag]

    return dict_gtr, gtr_to_id, id_to_gtr

def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3

def gazetteer_feature(s, gazetteer_list):
    if s.lower() in gazetteer_list:
        return gazetteer_list[s.lower()]
    else:
        return 0

def brown_feature(s, brown_dict):
    if s.lower() in brown_dict:
        return brown_dict[s.lower()]
    else:
        return 0

def prepare_sentence(str_words, word_to_id, char_to_id, gazetteer_list={}, brown_dict={}, lower=False):
    """
    Prepare a sentence for evaluation.
    """
    def f(x): return x.lower() if lower else x
    words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
             for w in str_words]
    chars = [[char_to_id[c] for c in w if c in char_to_id]
             for w in str_words]
    caps = [cap_feature(w) for w in str_words]
    gazetteer = [gazetteer_feature(w, gazetteer_list) for w in str_words]
    brown = [brown_feature(w, brown_dict) for w in str_words]
    return {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps,
        'gazetteer': gazetteer,
        'brown': brown
    }


def prepare_dataset(sentences, word_to_id, char_to_id, gazetteer_list, brown_dict, tag_to_id, l1_model, l1_f_eval, lower=False):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    def f(x): return x.lower() if lower else x
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
        caps = [cap_feature(w) for w in str_words]
        gazetteer = [gazetteer_feature(w, gazetteer_list) for w in str_words]
        brown = [brown_feature(w, brown_dict) for w in str_words]
        sent = {
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'caps': caps,
            'gazetteer': gazetteer,
            'brown': brown,
        }

        if l1_model is not None:
            input = create_input(sent, l1_model.parameters, False)
            try:
                if l1_model.parameters['crf']:
                    y_preds = np.array(f_eval(*input))[1:-1]
                else:
                    y_preds = f_eval(*input).argmax(axis=1)
                y_preds = [l1_model.id_to_tag[y_pred] for y_pred in y_preds]
            except Exception as e:
                y_preds = ["O"] * len(str_words)

            sent['pred'] = [0 if y_pred == "O" else 1 for y_pred in y_preds]

        tags = [tag_to_id[w[-1]] for w in s]
        sent['tags'] = tags
        data.append(sent)

    return data


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print 'Loading pretrained embeddings from %s...' % ext_emb_path
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8', errors='ignore')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word
