import jsonlines
from tqdm import tqdm
import multiprocessing
#把原文用/n分成很多局，存在一个列表
def main():
    data_list = []
    for line in tqdm(jsonlines.open('./raw_data_83650L_01-06.jsonl')):
        line['objects'][0]['text'] = [i for i in line['objects'][0]['text'].split('\n') if 30 < len(i) < 1500]
        data_list.append(line)

    with jsonlines.open("./format_data_83650L_01-06.jsonl", 'w') as w:
        for line in tqdm(data_list):
            w.write(line)


if __name__ == '__main__':
    main()