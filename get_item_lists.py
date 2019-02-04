import os

import unitypack

from pykoikatu import parse_token

in_dir = r'path\to\abdata\list\characustom'
out_dir = 'item_lists'


def parse_object(object, item_lists):
    data = object.read()
    print(data.name)
    assert all(x in '0123456789' for x in data.name.split('_')[-1])
    item_list_name = '_'.join(data.name.split('_')[:-1])
    if item_list_name not in item_lists:
        item_lists[item_list_name] = {}
    token, _ = parse_token(data.script, 0)
    for cols in token['dictList'].values():
        item_id = int(cols[0])
        item_name = cols[3]
        assert item_id not in item_lists[item_list_name]
        item_lists[item_list_name][item_id] = item_name


def parse_file(filename, item_lists):
    with open(filename, 'rb') as f:
        bundle = unitypack.load(f)
        for asset in bundle.assets:
            for _, object in asset.objects.items():
                if object.type == 'TextAsset':
                    parse_object(object, item_lists)


if __name__ == '__main__':
    item_lists = {}
    for root, dirs, files in os.walk(in_dir):
        for file in sorted(files):
            if not all(x in '0123456789' for x in file.split('.')[0]):
                continue
            filename = os.path.join(root, file)
            print(filename)
            parse_file(filename, item_lists)

    for item_list_name, item_list in item_lists.items():
        out_filename = os.path.join(out_dir, item_list_name + '.txt')
        with open(out_filename, 'w', encoding='utf-8') as f:
            for item_id, item_name in sorted(item_list.items()):
                f.write('{} {}\n'.format(item_id, item_name))
