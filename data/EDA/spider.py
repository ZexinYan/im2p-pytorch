import time
import urllib.request
import json
import multiprocessing
import os


def writer_proc(q):
    with open('./paragraphs_v1.json', 'r') as f:
        data = json.load(f)
    finished = [each[:-4] for each in os.listdir('./images')]
    unfinished = []
    for each in data:
        if str(each['image_id']) not in finished:
            unfinished.append({'image_id': each['image_id'], 'url': each['url']})
    i = 0
    while True:
        try:
            q.put(unfinished[i], block=False)
            i += 1
        except Exception as err:
            pass


def reader_proc(q):
    while True:
        try:
            info = q.get(block=False)
            try:
                urllib.request.urlretrieve(info['url'], './images/{}.jpg'.format(info['image_id']))
            except Exception as err:
                print(err)
                q.put(info)
                time.sleep(4)
        except Exception as err:
            pass

if __name__ == '__main__':
    q = multiprocessing.Queue(100)
    writer = multiprocessing.Process(target=writer_proc, args=(q,))
    writer.start()

    readers = []
    for i in range(20):
        readers.append(multiprocessing.Process(target=reader_proc, args=(q,)))
        readers[-1].start()

    writer.join()

    for each in readers:
        each.join()
