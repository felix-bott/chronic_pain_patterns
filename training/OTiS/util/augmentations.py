# Copyright (c) OTiS.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------


import torch
import torch.fft as fft

import numpy as np
from numbers import Real
from sklearn.utils import check_random_state

from typing import Any


class Rescaling(object):
    """
        Randomly rescale features of the sample.
    """
    def __init__(self, sigma=0.5) -> None:
        self.sigma = sigma

    def __call__(self, sample) -> Any:
        return sample * torch.normal(mean=torch.Tensor([1]), std=torch.Tensor([self.sigma]))

class Jitter(object):
    """
        Add gaussian noise to the sample.
    """
    def __init__(self, sigma=0.2, amplitude=0.6) -> None:
        self.sigma = sigma
        self.amplitude = amplitude

    def __call__(self, sample) -> Any:
        amplitude = self.amplitude * sample
        return sample + amplitude * torch.normal(mean=0, std=self.sigma, size=sample.shape)

class CropResizing(object):
    """
        Randomly crop the sample and resize to the original length.
    """
    def __init__(self, start_idx=None, lower_bnd=0.5, upper_bnd=1.0, fixed_crop_len=None, 
                 resize=True, fixed_resize_len=None) -> None:
        self.lower_bnd = lower_bnd
        self.upper_bnd = upper_bnd
        self.fixed_crop_len = fixed_crop_len
        self.start_idx = start_idx
        self.resize = resize
        self.fixed_resize_len = fixed_resize_len

    def __call__(self, sample) -> torch.Tensor:
        max_nb_time_steps = sample.shape[-1]
        if self.fixed_resize_len and self.fixed_resize_len < sample.shape[-1]:
            max_nb_time_steps = self.fixed_resize_len

        # define crop size
        crop_len = self.fixed_crop_len
        if self.fixed_crop_len is None:
            crop_len = int(max_nb_time_steps * np.random.uniform(low=self.lower_bnd, high=self.upper_bnd))
        
        # define cut-off point
        start_idx = self.start_idx
        if self.start_idx is None:
            try:
                start_idx = np.random.randint(low=0, high=sample.shape[-1] - crop_len)
            except ValueError:
                # if max_nb_time_steps - crop_len <= 0, np.random.randint() throws an error
                start_idx = 0

        # crop and resize the signal
        cropped_sample = sample[..., start_idx:start_idx+crop_len]
        if self.resize:
            resize_len = self.fixed_resize_len 
            if self.fixed_resize_len is None or self.fixed_resize_len > max_nb_time_steps:
                resize_len = max_nb_time_steps

            cropped_sample = torch.nn.functional.interpolate(cropped_sample, size=resize_len, mode="linear")

        return cropped_sample

class Masking(object):
    """
        Randomly zero-mask the sample.
        Got this from https://stackoverflow.com/questions/70092136/how-do-i-create-a-random-mask-matrix-where-we-mask-a-contiguous-length
        Don't touch the code!
    """
    def __init__(self, factor:float=0.75, fs:int=200, patch_size_sec:float=1, masking_mode="random", prob=1.0) -> None:
        self.factor = factor                    # fraction to be masked out
        self.patch_size = int(patch_size_sec * fs)   # patch_size[ticks] = patch_size[sec] * fs[Hz]
        self.masking_mode = masking_mode
        self.prob = prob

    def __call__(self, sample) -> Any:
        if np.random.uniform() < self.prob:
            # create the mask
            mask = torch.ones_like(sample)

            # determine the number of patches to be masked
            nb_patches = round(self.factor * sample.shape[-1] / self.patch_size)
            
            indices_weights = np.random.random((mask.shape[0], nb_patches + 1))

            number_of_ones = mask.shape[-1] - self.patch_size * nb_patches

            ones_sizes = np.round(indices_weights[:, :nb_patches].T
                                * (number_of_ones / np.sum(indices_weights, axis=-1))).T.astype(np.int32)
            ones_sizes[:, 1:] += self.patch_size

            zeros_start_indices = np.cumsum(ones_sizes, axis=-1)

            if self.masking_mode == "block":
                for zeros_idx in zeros_start_indices[0]:
                    mask[..., zeros_idx: zeros_idx + self.patch_size] = 0
            else:
                for sample_idx in range(len(mask)):
                    for zeros_idx in zeros_start_indices[sample_idx]:
                        mask[sample_idx, zeros_idx: zeros_idx + self.patch_size] = 0

            return sample * mask
        else:
            return sample
    
class FTSurrogate(object):
    """
    FT surrogate augmentation of a single EEG channel, as proposed in [1]_.
    Code (modified) from https://github.com/braindecode/braindecode/blob/master/braindecode/augmentation/functional.py 
    

    Parameters
    ----------
    X : torch.Tensor
        EEG input example.
    phase_noise_magnitude: float
        Float between 0 and 1 setting the range over which the phase
        pertubation is uniformly sampled:
        [0, `phase_noise_magnitude` * 2 * `pi`].
    channel_indep : bool
        Whether to sample phase perturbations independently for each channel or
        not. It is advised to set it to False when spatial information is
        important for the task, like in BCI.
    random_state: int | numpy.random.Generator, optional
        Used to draw the phase perturbation. Defaults to None.

    Returns
    -------
    torch.Tensor
        Transformed inputs.
    torch.Tensor
        Transformed labels.

    References
    ----------
    .. [1] Schwabedal, J. T., Snyder, J. C., Cakmak, A., Nemati, S., &
       Clifford, G. D. (2018). Addressing Class Imbalance in Classification
       Problems of Noisy Signals by using Fourier Transform Surrogates. arXiv
       preprint arXiv:1806.08675.
    """
    def __init__(self, phase_noise_magnitude, channel_indep=False, seed=None, prob=1.0) -> None:
        self.phase_noise_magnitude = phase_noise_magnitude
        self.channel_indep = channel_indep
        self.seed = seed
        self.prob = prob
        self._new_random_fft_phase = {
            0: self._new_random_fft_phase_even,
            1: self._new_random_fft_phase_odd
        }

    def _new_random_fft_phase_odd(self, c, n, device='cpu', seed=None):
        rng = check_random_state(seed)
        random_phase = torch.from_numpy(
            2j * np.pi * rng.random((c, (n - 1) // 2))
        ).to(device)

        return torch.cat([
            torch.zeros((c, 1), device=device),
            random_phase,
            -torch.flip(random_phase, [-1]).to(device=device)
        ], dim=-1)
    
    def _new_random_fft_phase_even(self, c, n, device='cpu', seed=None):
        rng = check_random_state(seed)
        random_phase = torch.from_numpy(
            2j * np.pi * rng.random((c, n // 2 - 1))
        ).to(device)

        return torch.cat([
            torch.zeros((c, 1), device=device),
            random_phase,
            torch.zeros((c, 1), device=device),
            -torch.flip(random_phase, [-1]).to(device=device)
        ], dim=-1)

    def __call__(self, sample) -> Any:
        if np.random.uniform() < self.prob:
            assert isinstance(
                self.phase_noise_magnitude,
                (Real, torch.FloatTensor, torch.cuda.FloatTensor)
            ) and 0 <= self.phase_noise_magnitude <= 1, (
                f"eps must be a float beween 0 and 1. Got {self.phase_noise_magnitude}."
            )

            f = fft.fft(sample.double(), dim=-1)

            n = f.shape[-1]
            random_phase = self._new_random_fft_phase[n % 2](
                f.shape[-2] if self.channel_indep else 1,
                n,
                device=sample.device,
                seed=self.seed
            )

            if not self.channel_indep:
                random_phase = torch.tile(random_phase, (f.shape[-2], 1))

            if isinstance(self.phase_noise_magnitude, torch.Tensor):
                self.phase_noise_magnitude = self.phase_noise_magnitude.to(sample.device)

            f_shifted = f * torch.exp(self.phase_noise_magnitude * random_phase)
            shifted = fft.ifft(f_shifted, dim=-1)
            sample_transformed = shifted.real.float()

            return sample_transformed
        else:
            return sample
    
