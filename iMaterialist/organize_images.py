import json
from os import path, listdir, mkdir, rename, remove

from PIL import Image

train_json_path = "/home/pbreton/.kaggle/competitions/imaterialist-challenge-furniture-2018/train.json"
test_json_path = "/home/pbreton/.kaggle/competitions/imaterialist-challenge-furniture-2018/test.json"
validation_json_path = "/home/pbreton/.kaggle/competitions/imaterialist-challenge-furniture-2018/validation.json"

def move_files_with_categories(dir):
    for file in listdir(dir):
        if not (path.isfile(path.join(dir, file)) and file.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']): continue
        category = file.split('.')[0].split('_')[-1]
        if not(path.exists(path.join(dir, category))): mkdir(path.join(dir, category))
        if not(path.exists(path.join(dir, category, file))): rename(path.join(dir, file), path.join(dir, category, file))

def check_no_truncated(dir):
    print('Checking %s' % dir)
    for file in listdir(dir):
        if path.isdir(path.join(dir, file)):
            check_no_truncated(path.join(dir, file))
        elif file.split('.')[-1].lower() == 'png':
            try:
                im = Image.open(path.join(dir, file))
                im.verify()
            except Exception:
                print('Image %s is truncated ! Removing it ...' % path.join(dir, file))
                remove(path.join(dir, file))


if __name__ == '__main__':
    # train = json.load(open(test_json_path))
    #
    # categories = set()
    # for file in listdir('./resources/train'):
    #     if path.isfile(path.join('./resources/train', file)):
    #         categories.add(file.split('.')[0].split('_')[1].lower())
    #
    #
    # for i in range(1, 129):
    #     if not(path.exists(path.join('./resources/train', str(i)))):  mkdir(path.join('./resources/train', str(i)))
    move_files_with_categories('./resources/train')
    check_no_truncated('./resources/train')