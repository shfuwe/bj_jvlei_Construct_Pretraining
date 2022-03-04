path='../11.8/'
def read_list(text_path):
    lsit=[]
    with open('%s' % text_path, 'r', encoding="utf8") as f:  # 打开一个文件只读模式
        line = f.readlines()  # 读取文件中的每一行，放入line列表中
        for line_list in line:
            lsit.append(line_list.replace('\n',''))
    return lsit
data_list=read_list( path+'data_list_clear_stop.txt')
print(len(data_list),len(data_list[0]))

def read_list(text_path):
    lsit=[]
    with open('%s' % text_path, 'r', encoding="utf8") as f:  # 打开一个文件只读模式
        line = f.readlines()  # 读取文件中的每一行，放入line列表中
        for line_list in line:
            lsit.append(line_list.replace('\n',''))
    return lsit
topic=read_list(path+'topic.txt')
print(len(topic),len(topic[0]))


from sklearn.cluster import KMeans
from sklearn import metrics

data_list_split=[i.split(' ') for i in data_list]
for i in range(len(data_list_split)):
    while '' in data_list_split[i]:
        data_list_split[i].remove('')
data_list_split[0]

import gensim.corpora as corpora
id2word = corpora.Dictionary(data_list_split)     # Create Dictionary
id2word.filter_extremes(no_below=3, no_above=0.5, keep_n=3000)


id2word.save_as_text(path+"dictionary")                   # save dict
texts = data_list_split                           # Create Corpus
corpus = [id2word.doc2bow(text) for text in texts]   # Term Document Frequency
print(corpus[:1])
print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

#词袋模型


print(len(id2word))
print(id2word)

from gensim import models

# 根据语料库和字典生成模型
tfidf_model = models.TfidfModel(corpus=corpus, dictionary=id2word)
tfidf_model.save('test_tfidf.model')  # 保存模型到本地
tfidf_model = models.TfidfModel.load('test_tfidf.model')  # 载入模型
corpus_tfidf = [tfidf_model[doc] for doc in corpus]


print(corpus_tfidf[:1])
print([[(id2word[id], freq) for id, freq in cp] for cp in corpus_tfidf[:1]])




#lda tf-idf 500
num_topics_=500
ldamodel_tfidf = models.LdaModel(corpus=corpus_tfidf, id2word=id2word, num_topics=num_topics_)
# 得到每条新闻的主题分布
topics_test_tfidf = ldamodel_tfidf.get_document_topics(corpus_tfidf)
ida_features_tfidf=[]
for i in topics_test_tfidf:
    new_feature=[0 for k in range(num_topics_)]
    for j in i:
        new_feature[j[0]]=j[1]
#         print(j)#(66, 0.17233049258408828)
#     print(new_feature)
    ida_features_tfidf.append(new_feature)
y_pred = KMeans(n_clusters=135, random_state=9).fit(ida_features_tfidf)
print("________________________________________________________________________________________________")
print("________________________________lda tf-idf 500________________________________")
print("________________________________________________________________________________________________")
print(metrics.adjusted_rand_score(topic, y_pred.labels_))
print(metrics.adjusted_mutual_info_score(topic, y_pred.labels_))


#lda tf-idf 1000
num_topics_=1000
ldamodel_tfidf = models.LdaModel(corpus=corpus_tfidf, id2word=id2word, num_topics=num_topics_)
# 得到每条新闻的主题分布
topics_test_tfidf = ldamodel_tfidf.get_document_topics(corpus_tfidf)
ida_features_tfidf=[]
for i in topics_test_tfidf:
    new_feature=[0 for k in range(num_topics_)]
    for j in i:
        new_feature[j[0]]=j[1]
#         print(j)#(66, 0.17233049258408828)
#     print(new_feature)
    ida_features_tfidf.append(new_feature)
y_pred = KMeans(n_clusters=135, random_state=9).fit(ida_features_tfidf)
print("________________________________________________________________________________________________")
print("________________________________lda tf-idf 1000________________________________")
print("________________________________________________________________________________________________")
print(metrics.adjusted_rand_score(topic, y_pred.labels_))
print(metrics.adjusted_mutual_info_score(topic, y_pred.labels_))




