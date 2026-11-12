import os
import scipy.io
from scipy.interpolate import interp1d
import numpy as np
import mne
from mne_features.feature_extraction import extract_features





import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "shared"))



def load_epochs(signals_path, norm_table, curr_subject, curr_set, curr_fold, splitSegments, upsample=False):

    # identify file corresponding to the current subject
    subject_file = None
    for file_name in os.listdir(signals_path):
        if file_name.startswith(f"{curr_subject}__"):
            subject_file = os.path.join(signals_path, file_name)
            break
    
    if subject_file is not None and os.path.exists(subject_file):
        try:
            dataSubject = scipy.io.loadmat(subject_file)
        except Exception as e:
            print(f"Error loading {subject_file}: {e}")
            return None
    else:
        print(f"File not found for subject: {curr_subject}")
        return None

    if 'brainSignals' in dataSubject:
        signals = dataSubject['brainSignals']
    else:
        signals = dataSubject['data']

    epochs_list, epochs_times = process_segments(signals, norm_table, curr_fold, curr_set, splitSegments, upsample)

    ch_names = [elem[0].item() for elem_idx, elem in enumerate(signals['label'][0,0].squeeze())]
    sfreq=int(1/np.mean(np.diff(epochs_times[0])))
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Stack all segments into a numpy array (n_epochs, n_channels, n_times)
    epochs_data = np.stack(epochs_list)

    # Create MNE Epochs object
    mne.set_log_level('WARNING')
    return mne.EpochsArray(epochs_data, info)


def process_segments(signals, norm_table, curr_fold, curr_set, splitSegments=True, upsample=False):
    epochs_list = []
    epochs_times = []
    epochs_start_times = []
    std_sgmt = []
    all_segment_time = signals['time'][0,0].squeeze()
    for iSegment in range(signals['trial'][0,0].size):
        segment = signals['trial'][0,0][0,iSegment]
        segment_time = all_segment_time[iSegment]

        if upsample and not upsample == 1:
            n_orig = segment.shape[1]
            n_new = n_orig * upsample
            t_orig = segment_time
            t_new = np.linspace(t_orig[0,0], t_orig[0,-1], n_new)
            f = interp1d(
                t_orig[0],
                segment,
                kind="linear",
                axis=1,
                fill_value="extrapolate"
            )
            segment = f(t_new)
            segment_time = t_new[np.newaxis, :]

        
        # demean signal (again)
        segment -= np.mean(segment, axis=1).reshape(-1, 1)

        # Rescale signal
        segment *= np.sqrt(np.exp(-9))

        # Normalize signals within study
        norm_factor = norm_table[curr_fold][(norm_table['set'] == curr_set)].values[0]
        segment *= norm_factor

        if splitSegments:
            k = 5  # Number of sub-segments
            segment_length = segment.shape[1] // k
            sub_segments = [segment[:, i * segment_length:(i + 1) * segment_length] for i in range(k)]
            sub_segments_time = [segment_time[0, i * segment_length:(i + 1) * segment_length] for i in range(k)]

            # Add all sub-segments to the epochs list
            epochs_list.extend(sub_segments)
            epochs_times.extend(sub_segments_time)
            epochs_start_times.extend([sub_segment_time[0] for sub_segment_time in sub_segments_time])
        else:
            # Segment shape: (n_channels, n_times) -> need (1, n_channels, n_times)
            epochs_list.append(segment)
            epochs_times.append(segment_time)


    if splitSegments:
        # Remove duplicate epochs based on start times
        unique_start_times = np.unique(np.round(epochs_start_times, decimals=6))
        filtered_epochs_list = []
        filtered_epochs_times = []

        for start_time in unique_start_times:
            indices = [i for i, t in enumerate(epochs_start_times) if abs(t - start_time) < 1e-6]
            # Keep only the first segment for this start time
            filtered_epochs_list.append(epochs_list[indices[0]])
            filtered_epochs_times.append(epochs_times[indices[0]])

        # Replace the original lists with the filtered ones
        epochs_list = filtered_epochs_list
        epochs_times = filtered_epochs_times

    return epochs_list, epochs_times


def compute_features_subject(epochs, njobs=1):
    # compute features (handcrafted)
    hc_selected_funcs = [
        'std',
        'kurtosis',
        'skewness',
        'quantile',
        'ptp_amp',
        'mean',
        'pow_freq_bands',
        'spect_entropy',
        'app_entropy',
        'samp_entropy', # all the above 12x10s segments ~ 25s
        
        'svd_entropy', # 12x10s segments ~ 40s
        'hurst_exp', # 12x10s segments ~ 40s
        'hjorth_complexity', # 12x10s segments ~ 14s
        'hjorth_mobility', # 12x10s segments ~ 8s
        'line_length', # 12x10s segments ~ 5s
        'wavelet_coef_energy', # 12x10s segments ~ 19s
        'higuchi_fd', # 12x10s segments ~ 8s
        'zero_crossings', # 12x10s segments ~ 9s
        'svd_fisher_info' # 12x10s segments ~ 40s
    ] # --> approx. 3min per subject

    hc_func_params = {
        'quantile__q': [0.1, 0.25, 0.75, 0.9],
        'pow_freq_bands__freq_bands': [0, 2, 4, 8, 13, 18, 24, 30, 49],
        'pow_freq_bands__ratios': 'all',
        'pow_freq_bands__ratios_triu': True,
        'pow_freq_bands__log': True,
        'pow_freq_bands__normalize': None,
    }

    features = extract_features(
        epochs.get_data(), epochs.info['sfreq'], hc_selected_funcs,
        funcs_params=hc_func_params, n_jobs=njobs, ch_names=epochs.ch_names,
        return_as_df=False)

    return features


def aggregate_features(X, func='mean', axis=0):
    if func == 'separate':
        return np.vstack(X)
    else:
        aggs = {'mean': np.nanmean, 'median': np.nanmedian}
        return np.vstack([aggs[func](x, axis=axis, keepdims=True) for x in X])


def residualize(y, z, B):
    """
    Subtract the component of y explained by confounder z, given fixed coefficients B.

    Parameters
    ----------
    y : 
        - Either a list of np.ndarray of shape (n_segments_i, n_features)  -> features case
        - Or array-like of shape (n_samples,) / (n_samples, n_features)    -> targets or already-aggregated features
    z : array-like, shape (n_samples,)
        Confounder values (e.g., age), one per subject/sample.
    B : np.ndarray
        Regression coefficients previously fitted on [1, z]:
        - shape (2,) or (2, 1) for scalar y
        - shape (2, n_features) for multivariate y

    Returns
    -------
    Residuals in the same structure as `y`:
        - list of arrays if `y` was list-of-arrays
        - np.ndarray or list of floats if `y` was 1D
    """
    z_arr = np.asarray(z, dtype=float)
    Z = np.column_stack([np.ones(len(z_arr)), z_arr])  # (n_samples, 2)

    # Case 1: y is list-of-arrays (features: one array per subject)
    if isinstance(y, list) and len(y) > 0 and isinstance(y[0], np.ndarray):
        # B should be (2, n_features)
        conf_effect = Z @ B  # (n_subjects, n_features)
        resid_list = []
        for Xi, ci in zip(y, conf_effect):
            resid_list.append(Xi - ci)  # broadcast over segments
        return resid_list

    # Case 2: y is numeric vector/matrix (targets or already-aggregated features)
    y_arr = np.asarray(y, dtype=float)  # (n_samples,) or (n_samples, n_features)
    y_hat = Z @ B                       # same leading dimension as y_arr
    resid = y_arr - y_hat

    # Preserve list-ness for 1D targets if input was a list
    if isinstance(y, list) and (y_arr.ndim == 1):
        return resid.tolist()
    else:
        return resid
