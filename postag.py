from transformers.pipelines import pipeline
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForTokenClassification
from transformers import logging
import sys

logging.set_verbosity_error()


def init_tagger():
    tokenizer = AutoTokenizer.from_pretrained(
        "gilf/french-camembert-postag-model", truncation=False)
    model = AutoModelForTokenClassification.from_pretrained(
        "gilf/french-camembert-postag-model")

    return pipeline(
        'ner', model=model, tokenizer=tokenizer,
        aggregation_strategy="simple")


def tag(data):
    # for elt in map(lambda x: x['entity_group'] + "\t" + x['word'],
    #                pos_tag(data)):
    #     print(elt)
    print({"sentence": data, "classes": list(map(lambda x: x,
                                                 pos_tag(data)))})


if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) == 0:
        print("usage postag <file>")

    else:
        try:
            f = open(args[0], "r")
            pos_tag = init_tagger()
            for line in f.readlines():
                tag(line)

        except FileNotFoundError:
            print("Input file \"" + args[0] + "\" not found..")
    # tag("40 grammes, de\t pommes")
