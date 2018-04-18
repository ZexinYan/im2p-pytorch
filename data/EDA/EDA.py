import json


if __name__ == '__main__':
    with open('../paragraphs_v1.json', 'r') as f:
        data = json.load(f)
    caption = dict()
    for each in data:
        caption[str(each['image_id'])] = {
            'paragraph': each['paragraph'],
            'url': each['url'],
            'image_id': each['image_id']
        }
    with open('../captions.json', 'w') as f:
        json.dump(caption, f)
    print(type(caption))
    print(caption['2406949'])
