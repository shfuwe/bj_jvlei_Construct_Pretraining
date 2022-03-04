# -*- coding: utf-8 -*-

from multiprocessing.dummy import Pool as ThreadPool
import json
import time

from tqdm import tqdm
import jsonlines
import requests as req

from urllib.parse import quote_plus

command = 'curl http://localhost:2222/rest/annotate --data-urlencode "text=President Obama called Wednesday on Congress to extend a tax break for students included in last year\'s economic stimulus package, arguing that the policy provides more generous assistance." --data "confidence=0.35" -H "Accept: application/json"'


def get_tags(text, url='http://localhost:2222/rest/annotate', con=0.5):
    url = 'http://localhost:2222/rest/annotate'
    url_encode = text
    resp = None
    # time.sleep(1)
    try:
        resp = req.get(f'{url}?text={quote_plus(url_encode)}&confidence={con}', headers={'Accept': 'application/json'})
    except Exception as e:
        print(e, text)
        return resp
    try:
        return resp.json()
    except:
        print('fuck')
        return {'false': True}


def doc_process(doc):
    title = doc['objects'][0]['title']
    content = doc['objects'][0]['text']
    tag_list = []
    text = title
    tags = get_tags(text=text)
    tag_list.append({'tags': tags})
    '<PARA>'.join(content[:20])

    for i in range(0, len(content), 20):
        tag_list.append({'tags': get_tags(text='<PARA>'.join(content[:10])[:6400])})
    doc['my_tags'] = tag_list
    return doc


if __name__ == '__main__':
    data_list = []
    counter = 0
    # total = []
    for line in tqdm(jsonlines.open('./format_data_83650L_01-06.jsonl')):
        if len(line['objects'][0]['text']) > 200:
            continue
            # print([len(i) for i in line['objects'][0]['text']])
            # print(line['topic'])
            # print(line)

        data_list.append(line)
        # total.extend([len("".join(line['objects'][0]['text']))])
        # if len(total) > 20000:
        #     print(json.dumps(line, ensure_ascii=False))
        counter += 1
        if counter >= 1000:
            break
    # print(sum(total) / len(total))
    # print(max(total))
    startTime = time.time()

    pool = ThreadPool(128)
    results = pool.map(doc_process, data_list)

    pool.close()
    pool.join()
    endTime = time.time()
    consumeTime = endTime - startTime
    print("程序运行时间：" + str(consumeTime) + " 秒")

    with jsonlines.open("./extracted_format_data_83650L_01-06.jsonl", 'w') as w:
        for doc in results:
            w.write(doc)
