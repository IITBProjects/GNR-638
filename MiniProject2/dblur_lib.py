import os
from torch.utils.data import DataLoader
from torchsummary import summary
from dblur.trainers.restormer import RestormerTrainer
from dblur.testers.restormer import RestormerTester
from dblur.testers.mscnn import MSCNNTester
from dblur.testers.stack_dmphn import StackDMPHNTester
from dblur.multi_modal_deblur import multi_modal_deblur


from dataset import VideoDataset
from utils import Utils

from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio


class DeblurLibrary:
    def __init__(self, config):
        self.config = config

    def create_dataset(self):
        dataset_config = self.config['dataset']
        train_image_dirs, test_image_dirs = Utils.train_test_split_dir(dataset_config['setA'], dataset_config['test_split'], dataset_config['num_dirs'])

        input_transform = lambda image: Utils.apply_gaussian(image, dataset_config['gaussian_filters'])
        self.train_dataset = VideoDataset(dataset_config['setA'], train_image_dirs, input_transform = input_transform)
        self.test_dataset = VideoDataset(dataset_config['setA'], test_image_dirs, input_transform = input_transform)

        print(f"Train size: {len(self.train_dataset)}, Test size: {len(self.test_dataset)}")


class Restormer(DeblurLibrary):
    def __init__(self, config):
        super().__init__(config)

    def train(self):
        train_config = self.config['train']

        train_dataloader = DataLoader(self.train_dataset, batch_size = train_config['batch_size'], shuffle = True)
        val_dataloader = DataLoader(self.test_dataset, batch_size = train_config['batch_size'], shuffle = False)

        restormer_trainer = RestormerTrainer()
        model = restormer_trainer.get_model(**train_config['restormer'])
        summary(model, (3, *self.config['dataset']['image_size']))
        optimizer = restormer_trainer.get_optimizer(model.parameters())
        loss_func = restormer_trainer.get_loss()
        lr_scheduler = restormer_trainer.get_lr_scheduler(optimizer)

        restormer_trainer.train(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            loss_func,
            save_checkpoint_freq=10,
            logs_folder='logs',
            checkpoint_save_name=train_config['model_save_path'],
            val_freq=20,
            write_logs=True,
            lr_scheduler=lr_scheduler,
            epochs=train_config['epochs']
        )

    def test(self):
        train_config = self.config['train']

        restormer_tester = RestormerTester()
        test_dataloader = DataLoader(self.test_dataset, batch_size = train_config['batch_size'], shuffle = False)
        model = restormer_tester.get_model(**train_config['restormer'])
        loss_func = restormer_tester.get_loss()

        restormer_tester.test(
            model,
            train_config['model_save_path'],
            test_dataloader,
            loss_func,
            is_checkpoint=True,
            window_slicing=True,
            window_size=256
        )

    def pred(self):
        train_config = self.config['train']

        restormer_tester = RestormerTester()
        model = restormer_tester.get_model(**train_config['restormer'])

        restormer_tester.deblur_imgs(
            model,
            train_config['model_save_path'],
            os.path.join(train_config['pred_dir'], 'blur'),
            os.path.join(train_config['pred_dir'], 'deblur_restormer'),
            is_checkpoint=True,
            batch_size=8,
            window_slicing=False
        )


class MultiModal:
    def __init__(self, config):
        super().__init__(config)

    def train(self):
        pass

    def test(self):
        pass
    
    def pred(self):
        pass
