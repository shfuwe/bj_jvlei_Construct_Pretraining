import jsonlines
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import argparse
from summa import keywords

stopwords = stopwords.words('english')


def keyword_extraction(text, num=20):
    return keywords.keywords(text=text, additional_stopwords=stopwords).split('\n')[:num]



def process(args):
    for line in jsonlines.open(args.raw_data_path):
        keywords = keyword_extraction(line['objects'][0]['title'] + line['objects'][0]['text'], num=args.n_keywords)
        print(keywords)
        input('-----------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocess')
    parser.add_argument('-raw_data_path',
                        default='/data/yuxiang/event_projs/Construct_Pretraining/data/raw_data/raw_data_83650L_01-06.jsonl',
                        type=str)

    parser.add_argument('-n_worker', default=8, type=int)
    parser.add_argument('-n_keywords', default=40, type=int)

    args = parser.parse_args()

    process(args)
