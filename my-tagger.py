#!/usr/bin/env python

import os
import time
import codecs
import optparse
import sys
import json
import numpy as np
from loader import prepare_sentence
from utils import create_input, iobes_iob, iob_ranges, zero_digits
from model import Model

optparser = optparse.OptionParser()
optparser.add_option(
    "-m", "--model", default="",
    help="Model location"
)
optparser.add_option(
    "-i", "--input", default="",
    help="Input file location"
)
optparser.add_option(
    "-o", "--output", default="",
    help="Output file location"
)
optparser.add_option(
    "-d", "--delimiter", default="__",
    help="Delimiter to separate words from their tags"
)
optparser.add_option(
    "--outputFormat", default="",
    help="Output file format"
)
opts = optparser.parse_args()[0]

# Check parameters validity
assert opts.delimiter
assert os.path.isdir(opts.model)
assert os.path.isfile(opts.input)

# Load existing model
print "Loading model..."
model = Model(model_path=opts.model)
parameters = model.parameters

# Load reverse mappings
word_to_id, char_to_id, tag_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
]

# Load the model
_, f_eval = model.build(training=False, **parameters)
model.reload()

# f_output = codecs.open(opts.output, 'w', 'utf-8')
start = time.time()

print 'Tagging...'
document = TextAnnotation(json_str=open(opts.input).read())
token_list = document.tokens
start = 0
count = 0

view_as_json = {}
cons_list = []

if 'NER_CONLL' in document.view_dictionary:
    del document.view_dictionary['NER_CONLL']

for sent_end_offset in document.sentences['sentenceEndPositions']:
    words_ini = token_list[start:sent_end_offset]
    line = " ".join(words_ini)
    if line:
        # Lowercase sentence
        if parameters['lower']:
            line = line.lower()
        # Replace all digits with zeros
        if parameters['zeros']:
            line = zero_digits(line)
        words = line.rstrip().split()
        # Prepare input
        sentence = prepare_sentence(words, word_to_id, char_to_id,
                                    lower=parameters['lower'])
        input = create_input(sentence, parameters, False)
        # Decoding
        if parameters['crf']:
            y_preds = np.array(f_eval(*input))[1:-1]
        else:
            y_preds = f_eval(*input).argmax(axis=1)
        y_preds = [model.id_to_tag[y_pred] for y_pred in y_preds]
        # Output tags in the IOB2 format
        if parameters['tag_scheme'] == 'iobes':
            y_preds = iobes_iob(y_preds)
        # Write tags
        assert len(y_preds) == len(words)
        assert len(y_preds) == len(words_ini)

        idx = 0
        while idx < len(y_preds):
            if y_preds[idx] == "O":
                idx += 1
            elif y_preds[idx].startswith("B-"):
                curr_label = y_preds[idx][2:]
                st = idx
                idx += 1
                while idx < len(y_preds) and y_preds[idx].startswith("I-"):
                    idx += 1
                cons_list.append({'start': start + st, 'end': start + idx, 'score': 1.0, 'label': curr_label})
            else:
                print("something wrong....")
                sys.exit(1)
        
    count += 1
    start = sent_end_offset + 1
    if count % 100 == 0:
        print count

view_as_json['viewData'] = [{'viewType': 'edu.illinois.cs.cogcomp.core.datastructures.textannotation.View', 'viewName': 'NER_CONLL', 'generator': 'my-lstm-crf-tagger', 'score': 1.0, 'constituents': cons_list}]

view_obj = View(view_as_json, document.get_tokens)

document.view_dictionary['NER_CONLL'] = view_obj

document_json = document.as_json

json.dump(docta_json, open(opts.output, "w"), indent=True)


print '---- %i lines tagged in %.4fs ----' % (count, time.time() - start)
# f_output.close()
