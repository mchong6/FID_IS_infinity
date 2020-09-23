import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from botorch.sampling.qmc import NormalQMCEngine
import numpy as np
import math
from sklearn.linear_model import LinearRegression
import math
import os
import glob
from tqdm import tqdm
from PIL import Image
from scipy import linalg 
from inception import *

class randn_sampler():
    """
    Generates z~N(0,1) using random sampling or scrambled Sobol sequences.
    Args:
        ndim: (int)
            The dimension of z.
        use_sobol: (bool)
            If True, sample z from scrambled Sobol sequence. Else, sample 
            from standard normal distribution.
            Default: False
        use_inv: (bool)
            If True, use inverse CDF to transform z from U[0,1] to N(0,1).
            Else, use Box-Muller transformation.
            Default: True
        cache: (bool)
            If True, we cache some amount of Sobol points and reorder them.
            This is mainly used for training GANs when we use two separate
            Sobol generators which helps stabilize the training.
            Default: False
            
    Examples::

        >>> sampler = randn_sampler(128, True)
        >>> z = sampler.draw(10) # Generates [10, 128] vector
    """

    def __init__(self, ndim, use_sobol=False, use_inv=True, cache=False):
        self.ndim = ndim
        self.cache = cache
        if use_sobol:
            self.sampler = NormalQMCEngine(d=ndim, inv_transform=use_inv)
            self.cached_points = torch.tensor([])
        else:
            self.sampler = None

    def draw(self, batch_size):
        if self.sampler is None:
            return torch.randn([batch_size, self.ndim])
        else:
            if self.cache:
                if len(self.cached_points) < batch_size:
                    # sample from sampler and reorder the points
                    self.cached_points = self.sampler.draw(int(1e6))[torch.randperm(int(1e6))]

                # Sample without replacement from cached points
                samples = self.cached_points[:batch_size]
                self.cached_points = self.cached_points[batch_size:]
                return samples
            else:
                return self.sampler.draw(batch_size)

def calculate_FID_infinity(gen_model, ndim, batch_size, gt_path, num_im=50000, num_points=15):
    """
    Calculates effectively unbiased FID_inf using extrapolation
    Args:
        gen_model: (nn.Module)
            The trained generator. Generator takes in z~N(0,1) and outputs
            an image of [-1, 1].
        ndim: (int)
            The dimension of z.
        batch_size: (int)
            The batch size of generator
        gt_path: (str)
            Path to saved FID statistics of true data.
        num_im: (int)
            Number of images we are generating to evaluate FID_inf.
            Default: 50000
        num_points: (int)
            Number of FID_N we evaluate to fit a line.
            Default: 15
    """
    # load pretrained inception model 
    inception_model = load_inception_net()

    # define a sobol_inv sampler
    z_sampler = randn_sampler(ndim, True)

    # get all activations of generated images
    activations, _ =  accumulate_activations(gen_model, inception_model, num_im, z_sampler, batch_size)

    fids = []

    # Choose the number of images to evaluate FID_N at regular intervals over N
    fid_batches = np.linspace(5000, num_im, num_points).astype('int32')

    # Evaluate FID_N
    for fid_batch_size in fid_batches:
        # sample with replacement
        np.random.shuffle(activations)
        fid_activations = activations[:fid_batch_size]
        fids.append(calculate_FID(inception_model, fid_activations, gt_path))
    fids = np.array(fids).reshape(-1, 1)
    
    # Fit linear regression
    reg = LinearRegression().fit(1/fid_batches.reshape(-1, 1), fids)
    fid_infinity = reg.predict(np.array([[0]]))[0,0]

    return fid_infinity

def calculate_FID_infinity_path(real_path, fake_path, batch_size=50, min_fake=5000, num_points=15):
    """
    Calculates effectively unbiased FID_inf using extrapolation given 
    paths to real and fake data
    Args:
        real_path: (str)
            Path to real dataset or precomputed .npz statistics.
        fake_path: (str)
            Path to fake dataset.
        batch_size: (int)
            The batch size for dataloader.
            Default: 50
        min_fake: (int)
            Minimum number of images to evaluate FID on.
            Default: 5000
        num_points: (int)
            Number of FID_N we evaluate to fit a line.
            Default: 15
    """
    # load pretrained inception model 
    inception_model = load_inception_net()

    # get all activations of generated images
    if real_path.endswith('.npz'):
        real_m, real_s = load_path_statistics(real_path)
    else:
        real_act, _ = compute_path_statistics(real_path, batch_size, model=inception_model)
        real_m, real_s = np.mean(real_act, axis=0), np.cov(real_act, rowvar=False)

    fake_act, _ = compute_path_statistics(fake_path, batch_size, model=inception_model)

    num_fake = len(fake_act)
    assert num_fake > min_fake, \
        'number of fake data must be greater than the minimum point for extrapolation'

    fids = []

    # Choose the number of images to evaluate FID_N at regular intervals over N
    fid_batches = np.linspace(min_fake, num_fake, num_points).astype('int32')

    # Evaluate FID_N
    for fid_batch_size in fid_batches:
        # sample with replacement
        np.random.shuffle(fake_act)
        fid_activations = fake_act[:fid_batch_size]
        m, s = np.mean(fid_activations, axis=0), np.cov(fid_activations, rowvar=False)
        FID = numpy_calculate_frechet_distance(m, s, real_m, real_s)
        fids.append(FID)
    fids = np.array(fids).reshape(-1, 1)
    
    # Fit linear regression
    reg = LinearRegression().fit(1/fid_batches.reshape(-1, 1), fids)
    fid_infinity = reg.predict(np.array([[0]]))[0,0]

    return fid_infinity

def calculate_IS_infinity(gen_model, ndim, batch_size, num_im=50000, num_points=15):
    """
    Calculates effectively unbiased IS_inf using extrapolation
    Args:
        gen_model: (nn.Module)
            The trained generator. Generator takes in z~N(0,1) and outputs
            an image of [-1, 1].
        ndim: (int)
            The dimension of z.
        batch_size: (int)
            The batch size of generator
        num_im: (int)
            Number of images we are generating to evaluate IS_inf.
            Default: 50000
        num_points: (int)
            Number of IS_N we evaluate to fit a line.
            Default: 15
    """
    # load pretrained inception model 
    inception_model = load_inception_net()

    # define a sobol_inv sampler
    z_sampler = randn_sampler(ndim, True)

    # get all activations of generated images
    _, logits =  accumulate_activations(gen_model, inception_model, num_im, z_sampler, batch_size)

    IS = []

    # Choose the number of images to evaluate IS_N at regular intervals over N
    IS_batches = np.linspace(5000, num_im, num_points).astype('int32')

    # Evaluate IS_N
    for IS_batch_size in IS_batches:
        # sample with replacement
        np.random.shuffle(logits)
        IS_logits = logits[:IS_batch_size]
        IS.append(calculate_inception_score(IS_logits)[0])
    IS = np.array(IS).reshape(-1, 1)
    
    # Fit linear regression
    reg = LinearRegression().fit(1/IS_batches.reshape(-1, 1), IS)
    IS_infinity = reg.predict(np.array([[0]]))[0,0]

    return IS_infinity

def calculate_IS_infinity_path(path, batch_size=50, min_fake=5000, num_points=15):
    """
    Calculates effectively unbiased IS_inf using extrapolation given 
    paths to real and fake data
    Args:
        path: (str)
            Path to fake dataset.
        batch_size: (int)
            The batch size for dataloader.
            Default: 50
        min_fake: (int)
            Minimum number of images to evaluate IS on.
            Default: 5000
        num_points: (int)
            Number of IS_N we evaluate to fit a line.
            Default: 15
    """
    # load pretrained inception model 
    inception_model = load_inception_net()

    # get all activations of generated images
    _, logits = compute_path_statistics(path, batch_size, model=inception_model)

    num_fake = len(logits)
    assert num_fake > min_fake, \
        'number of fake data must be greater than the minimum point for extrapolation'

    IS = []

    # Choose the number of images to evaluate FID_N at regular intervals over N
    IS_batches = np.linspace(min_fake, num_fake, num_points).astype('int32')

    # Evaluate IS_N
    for IS_batch_size in IS_batches:
        # sample with replacement
        np.random.shuffle(logits)
        IS_logits = logits[:IS_batch_size]
        IS.append(calculate_inception_score(IS_logits)[0])
    IS = np.array(IS).reshape(-1, 1)
    
    # Fit linear regression
    reg = LinearRegression().fit(1/IS_batches.reshape(-1, 1), IS)
    IS_infinity = reg.predict(np.array([[0]]))[0,0]

    return IS_infinity

################# Functions for calculating and saving dataset inception statistics ##################
class im_dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.imgpaths = self.get_imgpaths()
        
        self.transform = transforms.Compose([
                       transforms.Resize(64),
                       transforms.CenterCrop(64),
                       transforms.ToTensor()])

    def get_imgpaths(self):
        paths = glob.glob(os.path.join(self.data_dir, "**/*.jpg"), recursive=True) +\
            glob.glob(os.path.join(self.data_dir, "**/*.png"), recursive=True)
        return paths
    
    def __getitem__(self, idx):
        img_name = self.imgpaths[idx]
        image = self.transform(Image.open(img_name))
        return image

    def __len__(self):
        return len(self.imgpaths)

def load_path_statistics(path):
    """
    Given path to dataset npz file, load and return mu and sigma
    """
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
        return m, s
    else:
        raise RuntimeError('Invalid path: %s' % path)
        
def compute_path_statistics(path, batch_size, model=None):
    """
    Given path to a dataset, load and compute mu and sigma.
    """
    if not os.path.exists(path):
        raise RuntimeError('Invalid path: %s' % path)
        
    if model is None:
        model = load_inception_net()
    dataset = im_dataset(path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False)
    return get_activations(dataloader, model)

def get_activations(dataloader, model):
    """
    Get inception activations from dataset
    """
    pool = []
    logits = []

    for images in tqdm(dataloader):
        images = images.cuda()
        with torch.no_grad():
            pool_val, logits_val = model(images)
            pool += [pool_val]
            logits += [F.softmax(logits_val, 1)]

    return torch.cat(pool, 0).cpu().numpy(), torch.cat(logits, 0).cpu().numpy()

def accumulate_activations(gen_model, inception_model, num_im, z_sampler, batch_size):
    """
    Generate images and compute their Inception activations.
    """
    pool, logits = [], []
    for i in range(math.ceil(num_im/batch_size)):
        with torch.no_grad():
            z = z_sampler.draw(batch_size).cuda()
            fake_img = to_img(gen_model(z))

            pool_val, logits_val = inception_model(fake_img)
            pool += [pool_val]
            logits += [F.softmax(logits_val, 1)]

    pool =  torch.cat(pool, 0)[:num_im]
    logits = torch.cat(logits, 0)[:num_im]

    return pool.cpu().numpy(), logits.cpu().numpy()

def to_img(x):
    """
    Normalizes an image from [-1, 1] to [0, 1]
    """
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x



####################### Functions to help calculate FID and IS #######################
def calculate_FID(model, act, gt_npz):
    """
    calculate score given activations and path to npz
    """
    data_m, data_s = load_path_statistics(gt_npz)
    gen_m, gen_s = np.mean(act, axis=0), np.cov(act, rowvar=False)
    FID = numpy_calculate_frechet_distance(gen_m, gen_s, data_m, data_s)

    return FID

def calculate_inception_score(pred, num_splits=1):
    scores = []
    for index in range(num_splits):
        pred_chunk = pred[index * (pred.shape[0] // num_splits): (index + 1) * (pred.shape[0] // num_splits), :]
        kl_inception = pred_chunk * (np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
        kl_inception = np.mean(np.sum(kl_inception, 1))
        scores.append(np.exp(kl_inception))
    return np.mean(scores), np.std(scores)


def numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, required=True,
                        help=('Path to the dataset'))
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size to use')
    parser.add_argument('--out_path', type=str, required=True, 
                        help=('path to save dataset stats'))

    args = parser.parse_args()
                       
    act, logits = compute_path_statistics(args.path, args.batch_size)
    m, s = np.mean(act, axis=0), np.cov(act, rowvar=False)
    np.savez(args.out_path, mu=m, sigma=s)
