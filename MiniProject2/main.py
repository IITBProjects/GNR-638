import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
import os
from utils import Utils
from deblur import DeblurImages
from dblur_lib import Restormer
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import torch

config = json.load(open('config.json'))


def main():
    deblur_images = DeblurImages(config)
    deblur_images.create_datasets()
    # deblur_images.create_datasets_new()
    deblur_images.train()
    # deblur_images.pred()

    # restormer = Restormer(config)
    # restormer.create_dataset()
    # # restormer.train()
    # # restormer.test()
    # restormer.pred()

def script():
    # Utils.create_setA(config['dataset'], start_from = 200, end_at = 239)
    # Utils.create_setB(config['dataset'], start_from = 0)
    # total_images = sum([len(os.listdir(os.path.join(config['dataset']['setA'], dir))) for dir in os.listdir(config['dataset']['setA'])])
    # print(total_images)

    event_file = './logs/events.out.tfevents.1712706945.margav.675790.0'

    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    scalar_data = event_acc.Scalars('loss')

    for scalar_event in scalar_data:
        print(scalar_event.step, scalar_event.value)

if __name__ == '__main__':
    main()
    # script()
