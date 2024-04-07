import json
from utils import Utils
from deblur import DeblurImages

config = json.load(open('config.json'))


def main():
    deblur_images = DeblurImages(config)
    deblur_images.create_datasets()
    deblur_images.train()
    # x, y = deblur_images.train_dataset.__getitem__(0)
    # print(x.shape, y.shape, x.dtype, y.dtype)

def script():
    # Utils.create_setA(config['dataset'], start_from = 130)
    Utils.create_setB(config['dataset'], start_from = 0)

if __name__ == '__main__':
    main()
    # script()
