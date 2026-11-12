# Copyright (c) OTiS.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
from typing import Any, Tuple, Dict

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import util.augmentations as augmentations

import numpy as np
import scipy
import os
import re
import pandas as pd


import scipy.io
from multiprocessing import Pool
import time

import gc
import psutil



def print_memory(msg):
    process = psutil.Process(os.getpid())
    print(f"Memory usage {msg}: {process.memory_info().rss / 1024 ** 2:.2f} MB")

def process_file(args_tuple):
    currDir, currFile, currFreq, CVtable, sampleTypeSelect, args, NCtable = args_tuple

    # Extract subject ID
    patternSubject = r"(sub-[^_]+)"
    matchSubject = re.search(patternSubject, currFile)
    if not matchSubject:
        return None
    currSubject = matchSubject.group(1)

    # Extract set ID
    patternSet = r"(set-[^_]+)"
    matchSet = re.search(patternSet, currFile)
    if not matchSet:
        return None
    currSet = matchSet.group(1)

    # Check if the sample is valid based on CVtable
    if CVtable is not None:
        condition = (CVtable['participant_id'] == currSubject) & (CVtable['set'] == currSet[4:])
        if sum(condition) == 1:
            sampleType = CVtable[args.fold][condition].values[0]
        else:
            return None  # Skip this file

        if sampleType not in sampleTypeSelect:
            return None  # Skip this file

    # Load the MATLAB structure (.mat)
    try:
        dataSubject = scipy.io.loadmat(os.path.join(currDir, currFile))
    except Exception as e:
        print(f"Error loading {currFile}: {e}")
        return None  # Skip corrupted files

    # Extract signal data
    signal_data = []
    subject_list = []
    set_list = []
    freq_list = []

    demean_on = True
    # Check if 'brainSignals' is a field of dataSubject, else use 'dataset'
    if 'brainSignals' in dataSubject:
        key = 'brainSignals'
    elif 'data' in dataSubject:
        key = 'data'
    else:
        print(f"Neither 'brainSignals' nor 'data' found in {currFile}")
        return None
    for iSegment in range(dataSubject[key]['trial'][0,0].size):
        segment = dataSubject[key]['trial'][0,0][0,iSegment]

        # Check that signal has been demeaned
        if np.max(np.abs(np.mean(segment, axis=1) / np.std(segment, axis=1))) > 1e-5:
            if demean_on:
                segment -= np.mean(segment, axis=1).reshape(-1, 1)
            else:
                print(f"Signals not properly demeaned in {currFile}")
                return None

        # Rescale signal
        segment *= np.sqrt(np.exp(-9))

        # Normalize signals within study
        if NCtable[currFreq] is not None:
            norm_factor = NCtable[currFreq][args.fold][(NCtable[currFreq]['set'] == currSet[4:])].values[0]
            segment *= norm_factor

        # Normalize signals within segments
        if args.normalize_segments:
            segment /= np.mean(np.std(segment, axis=1))

        # In process_file:
        # Convert NumPy array to Torch tensor immediately
        if segment.dtype != np.float32:
            segment = segment.astype(np.float32)  # Ensure dtype compatibility

        if args.chop2patch_width_mult:
            # Chop signal length to be multiple of patch_width
            n_channels, n_times = segment.shape
            patch_width = args.patch_width
            new_n_times = (n_times // patch_width) * patch_width
            segment = segment[:, :new_n_times]

        tensor_segment = torch.from_numpy(segment)  # No memory copy if dtype matches

        # Store results
        signal_data.append(tensor_segment)
        subject_list.append(currSubject)
        set_list.append(currSet)
        freq_list.append(currFreq)

    return signal_data, subject_list, set_list, freq_list



class TimeSeriesDataset(Dataset):
    """
    Dataset for multi-domain time series analysis.
    """

    def __init__(self, data_path:str, labels_path:str=None, labels_mask_path:str=None, 
                 CVtable_path:str=None,
                 NCtable_path:str=None,
                 downstream_task:str=None, 
                 domain_offsets:Dict=None, domain_agnostic:str=False, 
                 univariate:bool=False, 
                 train:bool=False, 
                 test:bool=False,
                 N_val:int=1, 
                 args=None) -> None:
        """
            labels_path: path to labels (finetuning / online evaluation)
            labels_mask_path: path to labels masks (finetuning / online evaluation)
            downstream_task: downstream task (finetuning / online evaluation)

            domain_offsets: offsets for pos_embed_y
            domain_agnostic: share pos_embed_y across domains

            univariate: analyse each variate independently 
                        note - univariate analysis is domain agnostic

            train: training or validation / test
            N_val: nb of chunks to validate / test
        """
        print("hello 4")

        # Load data with function from scipy package #################################################################
        # List all files in the specified directory (data_path)
        if train:
            sampleTypeSelect = ['train'] # training
            if hasattr(args, 'combineTrainVal') and args.combineTrainVal:
                sampleTypeSelect = ['train', 'val']
        elif test:
            sampleTypeSelect = ['test'] # testing in indepedent data
        else:
            sampleTypeSelect = ['val'] # validating
        print(f"Loading {sampleTypeSelect} data from {data_path}")

        # load cross validation table
        if CVtable_path:
            CVtable = pd.read_csv(CVtable_path)
        else:
            CVtable = None


        # Parallel processing for loading data #######################################################################
        # Collect file names and their corresponding directories for multiple frequency bands
        allFiles = []
        allDirs = []
        allFreqs = []
        if('train' in sampleTypeSelect and args.train_freqs == 'all'):
            freq_all = ['all']
        elif('train' in sampleTypeSelect and args.train_freqs == 'multi'):
            freq_all = ['all', 'delta_special', 'theta', 'alpha', 'beta', 'gamma_broad']
        elif('train' in sampleTypeSelect and args.train_freqs == 'single'):
            freq_all = [args.freq_identifier]
        elif(not 'train' in sampleTypeSelect and args.test_freqs == 'all'):
            freq_all = ['all']
        elif(not 'train' in sampleTypeSelect and args.test_freqs == 'multi'):
            freq_all = ['all', 'delta_special', 'theta', 'alpha', 'beta', 'gamma_broad']
        elif(not 'train' in sampleTypeSelect and args.test_freqs == 'single'):
            freq_all = [args.freq_identifier]
        else:
            raise Exception("Invalid frequency selection")
            
        NCtable = {}
        for i_freq, freq in enumerate(freq_all):
            data_path_freq = data_path
            data_path_freq = data_path_freq.replace('freq-all', f'freq-{freq}')
            allFiles_freq = os.listdir(data_path_freq)                   
            if args.shrink_data and not freq == 'all':
                print("using only fraction of data")
                # Note: When excluding all files from the 'TDBRAIN' dataset memory capacitiy is not exceeded
                # # V1: exclude every third file of the 'TDBRAIN' dataset and include all others
                # V2: include every third file of the 'TDBRAIN' dataset and exclude every third file of the 'LEMON' dataset
                allFiles_freq = [f for i, f in enumerate(allFiles_freq) if (not i % 3 == 0 and 'LEMON' in f) or (i % 3 == 0 and 'TDBRAIN' in f)]
            else:
                print("using all available data")
            
            allFiles += allFiles_freq 
            allDirs += [data_path_freq for _ in range(len(allFiles_freq))]
            allFreqs += [freq for _ in range(len(allFiles_freq))]

            if NCtable_path:
                NCtable[freq] = pd.read_csv(NCtable_path.replace('_all_', f'_{freq}_'))
            else:
                NCtable[freq] = None

        # Prepare arguments for parallel execution
        process_args = [(d, f, fr, CVtable, sampleTypeSelect, args, NCtable) for d, f, fr in zip(allDirs, allFiles, allFreqs)]

        # load data from hard disk
        num_cpus = 16
        start_time = time.time()

        # # ---- parallel processing
        with Pool(processes=num_cpus) as pool:
            results = pool.map(process_file, process_args)
        # # ---- sequential processing

        # # ---- Run parallel processing in batches

        end_time = time.time()
        print(f"Time taken for parallel data loading: {end_time - start_time} seconds")

        # Flatten the results
        print_memory("start formatting")
        data, subjectList, setList, freqList = [], [], [], []
        for i, result in enumerate(results):

            if result:
                # Since data is already a torch tensor, directly append it
                for tensor_segment in result[0]:  
                    data.append(('eeg', tensor_segment))  # No conversion needed

                subjectList.extend(result[1])
                setList.extend(result[2])
                freqList.extend(result[3])

            results[i] = None
            
            if i % 100 == 0 and i != 0:
                print("formatting in progress")
                print_memory("during formatting")
                gc.collect()

        # Sequential processing for loading data #######################################################################
        # # Initialize empty lists to store combined data, subject IDs, and set IDs

        # # Check if 'eval' is a key in args, if not, set it to False

                

            
        #     # Iterate over all files in the directory

        #         # Extract the subject ID from the current file name using a regular expression
                
        #         # Extract the set ID from the current file name using a regular expression


        #                 continue

        #         # Load the MATLAB structure (.mat)

        #         # Iterate over all epochs (signal segments) in the current file
        #             # Check that signal has been demeaned
        #             if np.max(np.abs(np.mean(dataSubject['brainSignals']['trial'][0,0][0,iSegment], axis=1) \
        #                              / np.std(dataSubject['brainSignals']['trial'][0,0][0,iSegment], axis=1))) > 1e-5:
        #                              / np.std(dataSubject['brainSignals']['trial'][0,0][0,iSegment], axis=1))))

        #             # Rescale signal
        #             # Normalize signals within study
        #             # Normalize signals within segments (preserve variability across variates)
  

        #             # Append the current segment of brain signals to the dataList

        #             # Append the corresponding subject ID to the subjectList

        #             # Append the corresponding set ID to the setList

        #             # Append the corresponding frequency to the freqList
        

        # Convert the list of 2D NumPy arrays to a list of 2D PyTorch tensors
        # # LESS MEMORY EFFICIENT: #####

        # .unsqueeze(0) to add auxiliary channel (similar to rgb in imgs)
        domain = [(sample[0], sample[1].unsqueeze(0).shape) for sample in data]
        data = [sample[1].unsqueeze(0) for sample in data]

        self.univariate = univariate
        self.domain_agnostic = True if self.univariate else domain_agnostic

        self.domain = domain
        self.domains = {domain: shape for domain, shape in sorted(list(set(self.domain)))} # unique domains

        domain_list = [mod[0] for mod in domain]
        unique_domains = list(set(domain_list))
        
        self.domain_weights = {}
        for mod_current in unique_domains:
            mod_indices = torch.tensor([mod == mod_current for mod in domain_list])
            mod_weight = len(domain) / (len(unique_domains) * mod_indices.sum())
            self.domain_weights.update( {mod_current: mod_weight} )

        self.offsets = {}
        if domain_offsets is None:
            offset = 0
            for domain, shape in self.domains.items():
                self.offsets.update( {domain: offset} )
                if not self.domain_agnostic:
                    offset += shape[-2]
        else:
            self.offsets = domain_offsets

        # # As a sanity check, plot the signals of a few channels
        
        self.data = data
        self.MILgroup = subjectList
        self.freq = freqList

        if labels_path:
            # # Özgüns labels data load function ########

            # Load the .csv file with labels
            labels_df = pd.read_csv(labels_path)

            # Initialize a list to store the extracted painVars values
            painVarsList = []

            # Iterate over all signal segments of all subjects in all sets
            for i in range(len(subjectList)):
                # Extract the current subject ID and set ID
                currSubject = subjectList[i]
                currSet = setList[i]

                painVarValue = labels_df.loc[
                    (labels_df['participant_id'] == currSubject) & (labels_df['set'] == currSet[4:]), 
                    [args.fold+'_NORMpainVars', args.fold+'_NORMage']
                    ].values[0]

                # Append the painVarValue to the painVarsList
                painVarsList.append(painVarValue)
            
            if hasattr(args, "target_weights") and not len(args.target_weights) == args.upper_bnd - args.lower_bnd:
                raise Exception("mismatch between target_weights and number of targets")

            if hasattr(args, "nb_classes") and not args.nb_classes == args.upper_bnd - args.lower_bnd:
                raise Exception("mismatch between nb_classes and number of targets")


            # Convert painVarsList to a PyTorch tensor
            self.labels = torch.tensor(painVarsList, dtype=torch.float32)
            if(len(self.labels.shape) == 1):
                self.labels = self.labels.unsqueeze(1)
            

            # # Save observations and corresponding subject IDs


            #     # Save subjectList to a CSV file
 
        else:
            self.labels = torch.zeros(size=(len(self.data), max(1,args.upper_bnd)))

        if labels_mask_path:
            self.labels_mask = torch.load(labels_mask_path, map_location=torch.device('cpu')) # load to ram
        else:
            self.labels_mask = torch.ones_like(self.labels)

        self.downstream_task = downstream_task
        self.train = train 
        self.test = test or (not test and not train)
        self.N_val = N_val
        self.args = args

    def set_domain_offsets(self, domain_offsets:Dict=None):
        """set predefined domain offsets"""
        self.offsets = domain_offsets

    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """return a sample from the dataset at index idx"""
        # (1, V, T)
        data = self.data[idx]
        if self.univariate:
            # transform a multivariate time series with V variates into V univariate time series
            # (V, 1, T)
            data = data.permute(1, 0, 2)
        
        # number of samples to process
        N = data.shape[0]

        # (N, V, T) if multivariate analysis, 
        # (N*V, 1, T) else
        if self.train == False:
            # validate / test on more than one chunk per sample
            # Calculate number of overlapping chunks
            stride=32 
            if self.test == True:
                stride = 1
            
            N_val = max(int(((data.shape[-1] - self.args.time_steps) / stride) + 1), 1)
            if self.N_val != -1:
                N_val = min(N_val, self.N_val)
            N *= N_val

            # Create overlapping chunks
            data_chunks = [ data[..., i*stride:(i*stride+self.args.time_steps)] for i in range(N_val) ]

            # Concatenate chunks
            data = torch.cat(data_chunks, dim=0)
        else:
            N_val = max(data.shape[-1] - self.args.time_steps + 1, 1)
            if self.N_val != -1:
                N_val = min(N_val, self.N_val)
            N *= N_val

            transform = transforms.Compose([
                augmentations.CropResizing(fixed_resize_len=self.args.time_steps, 
                                           lower_bnd=self.args.crop_lower_bnd, 
                                           upper_bnd=self.args.crop_upper_bnd,
                                           resize=True),
                augmentations.FTSurrogate(phase_noise_magnitude=self.args.ft_surr_phase_noise, prob=0.5),
                augmentations.Jitter(sigma=self.args.jitter_sigma),
                augmentations.Rescaling(sigma=self.args.rescaling_sigma),
            ])
            
            # Create random chunks
            data_chunks = [ transform(data) for i in range(N_val) ]

            # Concatenate chunks
            data = torch.cat(data_chunks, dim=0)

        if self.downstream_task == 'regression':
            label = self.labels[idx][..., self.args.lower_bnd:self.args.upper_bnd]
            label_mask = self.labels_mask[idx][..., self.args.lower_bnd:self.args.upper_bnd]
        else:
            label = self.labels[idx].type(torch.LongTensor).argmax(dim=-1)
            label_mask = torch.ones_like(label)

        domain, _ = self.domain[idx]
        domain_offset = self.offsets[domain]

        label = [label for i in range(N)]
        label_mask = [label_mask for i in range(N)]
        domain = [domain for i in range(N)]
        
        return data, label, label_mask, self.args.patch_size, domain_offset, domain, self.args.time_steps, self.univariate, self.MILgroup[idx], self.freq[idx]

    @staticmethod
    def collate_fn(batch):
        # (p, q)
        patch_size = batch[0][3]
        grid_width = torch.tensor([sample[0].shape[-1] // patch_size[-1] for sample in batch])
        grid_height = torch.tensor([sample[0].shape[-2] // patch_size[-2] for sample in batch])

        # Determine the largest shape in the batch
        shape = [data.shape for sample in batch for data in sample[0]]
        max_values = [max(x) for x in zip(*shape)]
        max_variates = max_values[-2]
        max_timesteps = min(((max_values[-1] // patch_size[-1]) + 1) * patch_size[-1], batch[0][6]) # multiple of q 

        if grid_width.max() * patch_size[-1] < batch[0][6]:
            grid_width = grid_width + 1

        # Zero pad the input data to the largest shape
        # (B, 1, V_max, T_max)
        data = [torch.nn.functional.pad(data.unsqueeze(0), 
                                        pad=(0, int(max_timesteps - data.shape[-1]), 0, int(max_variates - data.shape[-2])), 
                                        mode="constant", value=0) for sample in batch for data in sample[0]]
        data = torch.stack(data, dim=0)

        # Create the attention mask 
        # (B, V'_max, T'_max), with V'_max=V_max/p, T'_max=T_max/p
        attn_mask = [torch.nn.functional.pad(torch.ones(size=(grid_height[idx], grid_width[idx])), 
                                                pad=(0, int(grid_width.max() - grid_width[idx]), 0, int(grid_height.max() - grid_height[idx])), 
                                                mode="constant", value=0) for idx, sample in enumerate(batch) for data in sample[0]]
        attn_mask = torch.stack(attn_mask, dim=0)
        
        # Create the pos embedding Y
        # (B, V'_max, T'_max)
        pos_embed_y = [torch.nn.functional.pad(torch.arange(grid_height[idx]).view(-1, 1).repeat(1, grid_width[idx]) + 1 + sample[4], 
                                                pad=(0, int(grid_width.max() - grid_width[idx]), 0, int(grid_height.max() - grid_height[idx])), 
                                                mode="constant", value=0) for idx, sample in enumerate(batch) for data in sample[0]]
        pos_embed_y = torch.stack(pos_embed_y, dim=0)

        # for convenience also return the labels here
        # (B, 1)
        label = torch.stack([label for sample in batch for label in sample[1]], dim=0)
        # (B, 1)
        label_mask = torch.stack([label_mask for sample in batch for label_mask in sample[2]], dim=0)

        domain = [domain for sample in batch for domain in sample[5]]
        MILgroup = [sample[-2] for sample in batch]
        freq = [sample[-1] for sample in batch]
        return data, attn_mask, torch.LongTensor(pos_embed_y), domain, MILgroup, label, label_mask, freq
    
    @staticmethod
    def collate_fn_ft(batch):
        # (B, 1, V, T)
        data = torch.stack([data.unsqueeze(0) for sample in batch for data in sample[0]], dim=0)
        # (B, 1)
        label = torch.stack([label for sample in batch for label in sample[1]], dim=0)
        # (B, 1)
        label_mask = torch.stack([label_mask for sample in batch for label_mask in sample[2]], dim=0)

        grid_width = batch[0][0].shape[-1] // batch[0][3][-1]
        grid_height = batch[0][0].shape[-2] // batch[0][3][-2]
        # (B, V', T'), V'=V/p and T'=T/q
        pos_embed_y = torch.arange(grid_height).view(-1, 1).repeat(len(batch), 1, grid_width) + 1 + batch[0][4]
        pos_embed_y = torch.stack([pos_embed_y[idx] for idx, sample in enumerate(batch) for data in sample[0]], dim=0)

        MILgroup = [sample[-2] for sample in batch]
        freq = [sample[-1] for sample in batch]
        return data, label, label_mask, torch.LongTensor(pos_embed_y), MILgroup, freq
