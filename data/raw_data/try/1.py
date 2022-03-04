import jsonlines
from tqdm import tqdm


def get_num():
    dic = {}
    for line in tqdm(jsonlines.open('../raw_data_83650L_01-06.jsonl')):
        if 'topic' in line:
            topic = line['topic']
            if len(topic) > 1:
                continue
            topic = topic[0]
            if topic not in dic:
                dic[topic] = 1
            else:
                dic[topic] += 1

    print(dic)
    nums_l = []
    for i in dic:
        nums_l.append(dic[i])
    nums_l.sort()
    store_list(nums_l, './nums_l.txt')

    dic_l = []
    for i in dic:
        if dic[i] > 1000:
            dic_l.append(i)
    store_list(dic_l, './ok_topic.txt')


def main():
    ok_topic = read_list('./ok_topic.txt')
    data_list = []
    topic_list = []
    for line in tqdm(jsonlines.open('../raw_data_83650L_01-06.jsonl')):
        if 'topic' in line:
            topic = line['topic']
            if len(topic) > 1:
                continue
            topic = topic[0]
            if topic not in ok_topic:
                continue
            text = line['objects'][0]['text']
            if len(text) < 100 or len(text) > 5000:
                print('skip')
                continue
            data_list.append(text.replace('\n', ' '))
            topic_list.append(ok_topic.index(topic))

    store_list(data_list, './text.txt')
    store_list(topic_list, './topic.txt')


def word_bag():
    stopwords = get_stop_words()
    data_list = read_list('./text.txt')
    data_list=data_list[0:10]
    print(data_list)

    dl = []
    for i in data_list:
        l = i.split(' ')
        dl.append(l)
    print(dl)
    # train_st_text = [drop_stopwords(stopwords, s) for s in dl]
    # print(train_st_text[:1])


def read_list(text_path):
    lsit = []
    with open('%s' % text_path, 'r', encoding="utf8") as f:  # 打开一个文件只读模式
        line = f.readlines()  # 读取文件中的每一行，放入line列表中
        for line_list in line:
            lsit.append(line_list.replace('\n', ''))
    return lsit


def store_list(lsit, text_path):
    ff = open(text_path, encoding='utf-8', mode='w')
    for line_list in lsit:
        ff.write(str(line_list))  # 写入一个新文件中
        ff.write("\n")


def get_stop_words():
    # 加载停用词
    stop_words_path = "../../stopwords.txt"
    stopwords = set([item.strip() for item in open(stop_words_path, 'r').readlines()])
    return stopwords
    # print(stopwords)


# 去掉文本中的停用词
def drop_stopwords(stopwords, line):
    line_clean = []
    for word in line:
        if word in stopwords:
            continue
        line_clean.append(word)
    return line_clean



if __name__ == '__main__':
    # print("??")

    # get_num()
    # main()
    word_bag()
