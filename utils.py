import torch
import dgl
from scipy.sparse import csc_matrix
import jsonlines


def load_data(data_path, remove_self_loop=True):
    news_list = [line for line in jsonlines.open(data_path) if len(line['selected_content']) > 0]
    title_list = [line['objects'][0]['title'] for line in news_list]
    paragraph_list = []
    paragraph_2_title = []
    for tid, line in enumerate(news_list):
        paragraph_list.extend(line['selected_content'])
        paragraph_2_title.extend([tid]*len(line['selected_content']))

    tag_list, keywords_list = [], []

    para2para = []
