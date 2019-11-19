# Effectively Unbiased FID and Inception Score and where to find them
This is the PyTorch implementation of Effectively Unbiased FID and Inception Score and where to find them. Note that since the scores are calculated with PyTorch, they are not directly comparable with the numbers obtained from TensorFlow. 

## Dependency
```bash
$ pip install -r requirements.txt
```

## How to use
First, generate the inception statistics for the groundtruth dataset to save time for future calculations. To do so, edit the dataloader in score_infinty.py to how you preprocess your data for your generator. Then run
```bash
$ python score_infinity.py --path path_to_dataset --out_path output_statistics.npz
```

### Evaluating FID infinity and IS infinity
To evaluate IS infinity or FID infinity from your evaluation script, add this following to your script

```python
from score_infinity import calculate_FID_infinity, calculate_IS_infinity

generator = load_your_generator()

FID_infinity = calculate_FID_infinity(generator, ndim, batch_size, gt_path=output_statistics.npz)
IS_infinity = calculate_IS_infinity(generator, ndim, batch_size)

```
This script assumes that your generator takes in a z~N(0,1) and outputs an image. It will use scrambled Sobol sequence with inverse CDF by default. Change z_sampler in each function if you wish to change the sampling method.

### Using Sobol sequence for training/evaluation GANs
```python
from score_infinity import randn_sampler

# For evaluating
sampler = randn_sampler(128, True)
z = sampler.draw(10) # Generates [10, 128] vector

# For training we need a separate sampler for gen/disc
# use_cache generates Sobol points and then randomly reorders them
gen_sampler = randn_sampler(128, True, use_cache=True)
disc_sampler = randn_sampler(128, True, use_cache=True)
```
Take a look at the comments in the code to understand the arguments it take. 

## Citation
If you use this code or ideas from our paper, please cite our paper:
```
@misc{chong2019effectively,
    title={Effectively Unbiased FID and Inception Score and where to find them},
    author={Min Jin Chong and David Forsyth},
    year={2019},
    eprint={1911.07023},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
```

## Acknowledgments
This code borrows heavily from [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch) and [pytorch-fid](https://github.com/mseitzer/pytorch-fid).
