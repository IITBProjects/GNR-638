import json
from vae_pipline import VAEPipeline
from vae_classifier import VAEClassifier


config = json.load(open('config.json'))

def main():
    # vae_pipeline = VAEPipeline(config)
    # vae_pipeline.create_dataset()
    # vae_pipeline.train()
    # vae_pipeline.visualise()

    vae_classifier = VAEClassifier(config)
    vae_classifier.create_dataset()
    vae_classifier.train()

if __name__ == '__main__':
    main()