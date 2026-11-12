# Compute inverse operator
from pathlib import Path
import mne
import mne
from pathlib import Path
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # CPP_code
from paths import CV_MAIN_TABLE, FINETUNE_DATA, INV_OPS_DIR, TEMPLATES_DIR, inputs_dir
from eeg_common import load_epochs
import pandas as pd
from joblib import Parallel, delayed


subjects_dir = Path(TEMPLATES_DIR)  # contains fsaverage/
bem_fpath = subjects_dir / "fsaverage" / "bem" / "fsaverage-ico4-bem-sol.fif"

signals_path = f"{FINETUNE_DATA}/sensorSignals/epoched_autoPreproc_CUT2.0_CUT0.5_250"
cv_table_path = CV_MAIN_TABLE
norm_table_path = inputs_dir("main") + "CV70-20-10Table_Sensor1020_all_MegaStrat_normFactorsNew.csv"
out_dir = Path(INV_OPS_DIR)

subject = "fsaverage"
njobs = 16


def compute_and_save_inverse_for_subject(epochs, out_fpath, src, bem, trans, montage):
    # Ensure EEG sensor locations exist in info
    epochs = epochs.copy().set_montage(montage, on_missing="ignore")

    # (Optional but common for EEG) add average reference projection
    epochs.set_eeg_reference("average", projection=True)

    # Forward model with template anatomy; excludes sources within 5 mm (mindist=5)
    fwd = mne.make_forward_solution(
        info=epochs.info,
        trans=trans,
        src=src,
        bem=bem,
        eeg=True, meg=False,
        mindist=5.0,
        n_jobs=1,
    )

    # Data-independent diagonal noise covariance (ad-hoc)
    noise_cov = mne.make_ad_hoc_cov(epochs.info)

    inv = mne.minimum_norm.make_inverse_operator(
        info=epochs.info,
        forward=fwd,
        noise_cov=noise_cov,
        # MNE's default values
        loose="auto",
        depth=0.8,
    )

    mne.minimum_norm.write_inverse_operator(out_fpath, inv, overwrite=True)

def process_subject(curr_subject, curr_set, trans, montage, drop_chan):
    epochs = load_epochs(
        signals_path, norm_table,
        curr_subject, curr_set, curr_fold, splitSegments
    )
    epochs = epochs.drop_channels(drop_chan)

    out_dir_fold = out_dir / curr_fold
    out_dir_fold.mkdir(parents=True, exist_ok=True)
    out_fpath = out_dir_fold / f"{curr_subject}-inv.fif"

    compute_and_save_inverse_for_subject(epochs, out_fpath, src, bem, trans, montage)
    return out_fpath


# 4098 dipoles / hemisphere  -> oct6
src = mne.setup_source_space(subject, spacing="oct6",
                             subjects_dir=subjects_dir, add_dist=False)

if not bem_fpath.exists():
    print("computing bem")
    bem_model = mne.make_bem_model(
        subject="fsaverage",
        ico=4,
        subjects_dir=subjects_dir
    )
    bem = mne.make_bem_solution(bem_model)
    mne.write_bem_solution(bem_fpath, bem)
else:
    print("loading existing bem")
    bem = mne.read_bem_solution(bem_fpath)

# fsaverage transform (shipped with fsaverage dataset)
trans = "fsaverage"

# your montage choice (must match channel names in your .mat)
montage = mne.channels.make_standard_montage("standard_1005")

splitSegments = False
drop_chan = ['LE', 'RE']
cv_table = pd.read_csv(cv_table_path)
norm_table = pd.read_csv(norm_table_path)

subjects = cv_table['participant_id']
sets = cv_table['set']

selected_folds = ['Fold2', 'Fold3', 'Fold4', 'Fold5', 'Fold6', 'Fold7', 'Fold8', 'Fold9', 'Fold10'] # ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5', 'Fold6', 'Fold7', 'Fold8', 'Fold9', 'Fold10']

for curr_fold in selected_folds:
    print(f"computing inv ops for {curr_fold}")
    Parallel(n_jobs=njobs)(
        delayed(process_subject)(curr_subject, curr_set, trans, montage, drop_chan)
        for curr_subject, curr_set in zip(subjects, sets)
    )
