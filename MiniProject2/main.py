import json
import os
from utils import Utils
from deblur import DeblurImages
from dblur_lib import Restormer

config = json.load(open('config.json'))


def main():
    deblur_images = DeblurImages(config)
    deblur_images.create_datasets()
    deblur_images.train()

    # restormer = Restormer(config)
    # restormer.create_dataset()
    # restormer.train()

def script():
    # Utils.create_setA(config['dataset'], start_from = 18, end_at = 39)
    # Utils.create_setB(config['dataset'], start_from = 0)
    total_images = sum([len(os.listdir(os.path.join(config['dataset']['setA'], dir))) for dir in os.listdir(config['dataset']['setA'])])
    print(total_images)

if __name__ == '__main__':
    main()
    # script()
