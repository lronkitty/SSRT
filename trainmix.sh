conda activate ssrt
python main.py trainer=ssrt optim=ssrt model=ssrt gpu_ids=\'0\' data=icvl  data.bs=2 noise=mixture trainer.params.num_sanity_val_steps=0 model.params.channels=48 test=icvl_mix