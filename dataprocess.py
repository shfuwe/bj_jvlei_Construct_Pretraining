import jsonlines
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from multiprocessing.dummy import Pool as ThreadPool
import time

from tqdm import tqdm
import argparse
# from summa import keywords
import summa

stopwords = stopwords.words('english')
with open('./data/stopwords.txt', 'r') as f:
    for line in f.readlines():
        stopwords.append(line.strip())

# pbar = tqdm()

def keyword_extraction(text, num=20):
    words = summa.keywords.keywords(text=text, additional_stopwords=stopwords).split('\n')
    return words[:num]


def process_one(line):
    text = line['objects'][0]['title'] + line['objects'][0]['text']
    tags = [i['label'] for i in line['objects'][0]['tags']]
    # print(len(line['objects'][0]['text']) )
    keywords = keyword_extraction(text, num=args.n_keywords)
    contents = []
    for sent in line['objects'][0]['text'].split('\n'):
        for word in keywords + tags:
            if word in sent:
                contents.append(sent)
                break
    # print(len(contents), len(line['objects'][0]['text'].split('\n')))

    line['keywords'] = keywords
    line['selected_content'] = contents
    # pbar.update(1)
    return line


def process(args):
    data = [line for line in tqdm(jsonlines.open(args.raw_data_path)) if
            len(line['objects'][0]['title'] + line['objects'][0]['text']) < 50000]

    startTime = time.time()

    pool = ThreadPool(8)
    results = pool.map(process_one, data)
    endTime = time.time()
    consumeTime = endTime - startTime
    print("程序运行时间：" + str(consumeTime) + " 秒")
    with jsonlines.open("./data/raw_data/processed_format_data_83650L_01-06.jsonl", 'w') as w:
        for doc in results:
            w.write(doc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocess')
    parser.add_argument('-raw_data_path',
                        default='/data/yuxiang/event_projs/Construct_Pretraining/data/raw_data/raw_data_83650L_01-06.jsonl',
                        type=str)

    parser.add_argument('-n_worker', default=8, type=int)
    parser.add_argument('-n_keywords', default=20, type=int)
    args = parser.parse_args()

    process(args)
