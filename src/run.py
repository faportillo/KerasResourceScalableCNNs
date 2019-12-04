import config as cfg
from train_model import train_model
from prune_model import prune_model

def main():
    if cfg.model_train:
        train_model()
    if cfg.model_prune:
        prune_model()

if __name__ == '__main__':
    main()
