# [Deep learning for undersampled MRI reconstruction](https://arxiv.org/pdf/1709.02576.pdf)
![alt text](https://github.com/hpkim0512/Unet/blob/master/web/img/architecture.png)

# Prerequisites
- Python 3.5
- [Tensorflow 1.3](https://www.tensorflow.org/)
- [h5py](www.h5py.org/)

# Usage
To train a model with dataset Version 7.3 MAT-file 'data_set.mat':

    ... add MAT-file to data directory  => ./data/data_set.mat ...
    $ python main.py --data_set=data_set

'dataset.mat' have to consists of 'input' and 'label' whose shape as [width, height, num_of_data]

![alt text](https://github.com/hpkim0512/Unet/blob/master/web/img/matfile_format.JPG)

If you have more than one GPU, it supports to activate multi-GPUs:

    $ python main.py --data_set=data_set --num_gpu=4

To test with an existing model (./logs/model) and test_set Version 7.3 MAT-file 'test_set.mat':

    ... add MAT-file to data directory  => ./data/test_set.mat ...
    $ python main.py --is_train=False --ckpt_dir=model --test_set=test_set
