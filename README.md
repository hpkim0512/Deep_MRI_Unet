# [Deep learning for undersampled MRI reconstruction](https://arxiv.org/pdf/1709.02576.pdf)
![alt text](https://github.com/hpkim0512/Unet/blob/master/web/img/architecture.png)
This paper presents a deep learning method for faster magnetic resonance imaging (MRI) by reducing k-space data with sub-Nyquist sampling strategies and provides a rationale for why the proposed approach works well.
Uniform subsampling is used in the time-consuming phase-encoding direction to capture high-resolution image information, while permitting the image-folding problem dictated by the Poisson summation formula.
To deal with the localization uncertainty due to image folding, very few low-frequency k-space data are added.
Training the deep learning net involves input and output images that are pairs of Fourier transforms of the subsampled and fully sampled k-space data.
Numerous experiments show the remarkable performance of the proposed method; only 29% of k-space data can generate images of high quality as effectively as standard MRI reconstruction with fully sampled data.

# Prerequisites
- Python 3.5
- [Tensorflow 1.3](https://www.tensorflow.org/)
- [h5py](www.h5py.org/)

# Usage
To train a model with dataset Version 7.3 MAT-file 'dataset.mat':

    ... add MAT-file to data directory  => ./data/dataset.mat ...
    $ python main.py --data_set=dataset

'dataset.mat' have to consists of 'input' and 'label' whose shape as [width, height, num_of_data]

![alt text](https://github.com/hpkim0512/Unet/blob/master/web/img/matfile_format.JPG)

If you have more than one GPU, it supports to activate multi-GPUs:

    $ python main.py --data_set=dataset --num_gpu=4

To test with an existing model (./logs/model) and testset Version 7.3 MAT-file 'testset.mat':

    ... add MAT-file to data directory  => ./data/testset.mat ...
    $ python main.py --is_train=False --ckpt_dir=model --test_set=testset
