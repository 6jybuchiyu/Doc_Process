import os

import mne
from matplotlib import pyplot as plt


def run_ica(raw, ICA_folder):
    ica = mne.preprocessing.ICA(n_components=20)
    ica.fit(raw, reject_by_annotation=False)

    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=raw.info['ch_names'][0], reject_by_annotation=False)
    mus_indices, mus_scores = ica.find_bads_muscle(raw)
    artifacts_indices = eog_indices + mus_indices
    ica.exclude = artifacts_indices
    raw = ica.apply(raw, exclude=ica.exclude)

    ica.plot_components(nrows=5, ncols=4, show=False)
    plt.savefig(os.path.join(ICA_folder, "plot_components.jpg"))

    for i in range(ica.n_components):
        ica.plot_properties(raw, picks=[i], show=False)
        plt.savefig(os.path.join(ICA_folder, f"ICA_{i}.jpg"))

    return raw
