from torch.utils.data import DataLoader
from dblur.trainers.restormer import RestormerTrainer
from dblur.testers.restormer import RestormerTester
from dataset import RuntimeImageDataset
from utils import Utils

class Restormer:
    def __init__(self, config):
        self.config = config

    def create_dataset(self):
        dataset_config = self.config['dataset']        
        train_image_paths, test_image_paths = Utils.train_test_split(dataset_config['setA'], dataset_config['test_split'], dataset_config['num_dirs'])

        input_transform = lambda image: Utils.apply_gaussian(image, dataset_config['gaussian_filters'])
        self.train_dataset = RuntimeImageDataset(dataset_config['setA'], train_image_paths, input_transform = input_transform)
        self.test_dataset = RuntimeImageDataset(dataset_config['setA'], test_image_paths, input_transform = input_transform)

        print(f"Train size: {len(self.train_dataset)}, Test size: {len(self.test_dataset)}")

    def train(self):
        train_config = self.config['train']

        train_dataloader = DataLoader(self.train_dataset, batch_size = train_config['batch_size'], shuffle = True)
        val_dataloader = DataLoader(self.test_dataset, batch_size = train_config['batch_size'], shuffle = False)

        restormer_trainer = RestormerTrainer()
        self.model = restormer_trainer.get_model(num_layers=4, num_refinement_blocks = 2)
        optimizer = restormer_trainer.get_optimizer(self.model.parameters())
        loss_func = restormer_trainer.get_loss()
        lr_scheduler = restormer_trainer.get_lr_scheduler(optimizer)

        restormer_trainer.train(
            self.model,
            train_dataloader,
            val_dataloader,
            optimizer,
            loss_func,
            save_checkpoint_freq=10,
            logs_folder='logs',
            checkpoint_save_name=train_config['model_save_path'],
            val_freq=100,
            write_logs=True,
            lr_scheduler=lr_scheduler,
            epochs=train_config['epochs']
        )

    # def test(self):
    #     restormer_tester = RestormerTester()
    #     test_dataloader = restormer_tester.get_test_dataloader(test_img_dir, batch_size=8)
    #     model = restormer_tester.get_model(num_layers = 4, num_refinement_blocks = 2)
    #     loss_func = restormer_tester.get_loss()

    #     # test model on test dataset.
    #     restormer_tester.test(model,
    #                         model_path,
    #                         test_dataloader,
    #                         loss_func,
    #                         is_checkpoint=True,
    #                         window_slicing=True,
    #                         window_size=256)
                            
    #     # deblur images in a directory using pretrained model
    #     restormer_tester.deblur_imgs(model,
    #                                 model_path,
    #                                 blur_img_dir,
    #                                 sharp_img_dir,
    #                                 is_checkpoint=True,
    #                                 batch_size=8,
    #                                 window_slicing=False)

    #     #deblur single image using pretrained model
    #     restormer_tester.deblur_single_img(model,
    #                                     model_path,
    #                                     blur_img_path,
    #                                     sharp_img_path,
    #                                     is_checkpoint=True,
    #                                     window_slicing=False)