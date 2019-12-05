import config as cfg

def main():
    if cfg.model_train:
        from train_model import train_model
        train_model()
    if cfg.model_eval:
        from eval_model import eval_model
        eval_model()
    if cfg.model_prune:
        from prune_model import prune_model
        prune_model()

if __name__ == '__main__':
    main()
