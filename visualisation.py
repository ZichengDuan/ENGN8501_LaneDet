import os
import cv2
import json
from matplotlib import pyplot as plt

def load_json(root, file_dir, file_name):
    results = []
    with open(os.path.join(root, file_dir, file_name), 'r') as f:
         results += [json.loads(x.strip()) for x in f.readlines()]
    return results






if __name__ == "__main__":
    root = '/root/datasets/llamas'
    res = load_json(root, 'bezier_labels', 'train_3.json')
    file_name = res[0]['raw_file'].split('.')[0] + '.png'
    img = cv2.imread(os.path.join(root, file_name))
    plt.imshow(img)
    plt.show()