python main.py trainer=ssrt optim=ssrt model=ssrt gpu_ids=\'0\' data=icvl  data.bs=2 noise.params.sigma_max=95 trainer.params.num_sanity_val_steps=0 model.params.channels=48 test=icvl95