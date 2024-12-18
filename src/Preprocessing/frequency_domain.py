import pandas as pd
import random
import numpy as np
import glob
import itertools
import matplotlib.pyplot as plt
from scipy import signal


class Feature:
    def group_into_bands(fft_data, fft_frequency, band_edges):
        # Map each frequency to its band index
        band_indices = np.digitize(fft_frequency, band_edges)
        mean_vals, std_vals = [], []
        # Compute mean/std in each band
        for band_idx in range(1, len(band_edges)):
            band_mask = (band_indices == band_idx)
            band_data = fft_data[band_mask]
            if len(band_data) == 0:
                mean_vals.append(0.0)
                std_vals.append(0.0)
            else:
                mean_vals.append(np.mean(band_data))
                std_vals.append(np.std(band_data))
        return mean_vals, std_vals

    def fft_feature_extraction(slices, sampling_freq, bands=[0.1, 4, 8, 12, 30, 50, 70, 100, 180]):
        # Extract features (log-amplitude means & stds) for each slice
        all_features = []
        for slice_data in slices:
            channels = slice_data.shape[0]
            slice_features = []
            for ch in range(channels):
                data = slice_data[ch, :]
                fft_vals = np.fft.rfft(data)
                fft_data_log = np.log10(np.abs(fft_vals) + 1e-10)  # log amplitude
                fft_freq = np.fft.rfftfreq(n=data.shape[-1], d=1.0 / sampling_freq)
                mean_vals, std_vals = slices.group_into_bands(fft_data_log, fft_freq, bands)
                ch_features = np.concatenate([mean_vals, std_vals], axis=0)
                slice_features.append(ch_features)
            all_features.append(np.array(slice_features))
        return all_features