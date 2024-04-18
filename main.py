import logging
import hydra
from omegaconf import DictConfig
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@hydra.main(config_path="ssrt/config", config_name="config")
def main(cfg: DictConfig) -> None: 
    try:
        os.environ["CUDA_VISIBLE_DEVICES"]=cfg.gpu_ids
    except:
         os.environ["CUDA_VISIBLE_DEVICES"]=str(cfg.gpu_ids)
    import torch
    cfg.trainer.params.gpus = torch.cuda.device_count()
    from ssrt.train import train
    from ssrt.test import test
    logger.info(f"Current working directory : {os.getcwd()}")

    # try:
    
    if cfg.mode.name == "train":
        train(cfg)
    else:
        test(cfg)
    # except Exception as e:
    #     logger.critical(e, exc_info=True)


if __name__ == "__main__":
           main()
