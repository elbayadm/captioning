# Captioning code in PyTorch
Building from [Ruotian Luo's code for captioning](https://github.com/ruotianluo/ImageCaptioning.pytorch)

### Data preprocessing:

    > python scripts/prepare/prepro.py 


### Training:

Training requires some directories for saving the model's snapshots, the tensorboard events 

    > mkdir -p save events

To train a model under the parameters defined in config.yaml

    > python train.py -c config.yaml 

Check **options/opts.py** for more about the options.

To evaluate a model:

    > python eval.py -c config


To submit jobs via OAR use either _train.sh_ or _select_train.sh_
