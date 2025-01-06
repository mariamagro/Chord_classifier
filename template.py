# -*- coding: utf-8 -*-
"""
LAB TEMPLATE

Audio features with SMOTE and feature correlation analysis
"""

# Generic imports
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
# Import ML tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
# Import audio feature library
import librosa as lbs
# Import our custom lib for audio feature extraction (makes use of librosa)
import audio_features as af
import pandas as pd
from scipy.stats import skew, kurtosis


##############################################################################
# Feature extraction function
##############################################################################

def extract_features(X, verbose=True):
    """
    Extracts a feature matrix for the input data X.
    """

    # These parameters were already given
    num_data = len(X)
    sr = lbs.get_samplerate(X[0])
    n_feat = 87 # Adjusted the number of parameters to the correct one
    M = np.zeros((num_data, n_feat))

    # Parameters that will be used in the extraction of features of the audio
    # All of this have been tuned using grid search
    flen = 1024
    nsub = 10
    hop = 64
    thr_db = 20
    n_mfcc = 13

    for i in range(num_data):
        if verbose:
            print('%d/%d... ' % (i + 1, num_data), end='')

        # Initial call and preprocess of the audio
        audio_data, _ = lbs.load(X[i], sr=sr)
        audio_data = af.preprocess_audio(audio_data, thr=thr_db)

        # Extraction of the entropies using the function given in the audio features code
        energy_entropies = af.get_energy_entropy(audio_data, flen=flen, hop=hop, nsub=nsub)

        # The gathered statistics are the mean and the standard deviation
        M[i, 0] = np.nanmean(energy_entropies)
        M[i, 1] = np.nanmax(energy_entropies)

        # Extraction of the MFCC features using librosa library, 13 components will be extracted
        mfccs = lbs.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, hop_length=hop, n_fft=flen)

        # This has proven to be very useful so many statistics will be drawn, the correlation will be later analyzed

        # First obtain the mean and standard deviation
        for j in range(13):
            M[i, 2 + j] = np.nanmean(mfccs[j, :])
            M[i, 15 + j] = np.nanstd(mfccs[j, :])

        # Obtain the first derivative of the coefficients using librosa library
        delta_mfccs = lbs.feature.delta(mfccs)
        for j in range(13):
            # Obtain the means
            M[i, 28 + j] = np.nanmean(delta_mfccs[j, :])

        # Obtain the second derivative (changes of curvature)
        delta2_mfccs = lbs.feature.delta(mfccs, order=2)
        for j in range(13):
            # Obtain also the mean
            M[i, 41 + j] = np.nanmean(delta2_mfccs[j, :])

        # Obtain also the skewness and kurtosis of the coefficients
        for j in range(13):
            M[i, 54 + j] = skew(mfccs[j, :])
            M[i, 67 + j] = kurtosis(mfccs[j, :])

        # Get the spectral entropies using the function in the audio features
        spectral_entropies = af.get_spectral_entropy(audio_data, flen=flen, hop=hop, nsub=nsub)
        # Obtain the mean
        M[i, 80] = np.nanmean(spectral_entropies)

        # Get the spectral centroids using the function in the audio features
        spectral_centroids = af.get_spectral_centroid(audio_data, sr, flen=flen, hop=hop)
        # Obtain the mean
        M[i, 81] = np.nanmean(spectral_centroids)

        # Get the harmonic ratio using the function in the audio features
        harmonic_ratios = af.get_harmonic_ratio(audio_data, sr, flen=flen, hop=hop)
        # Obtain the mean
        M[i, 82] = np.nanmean(harmonic_ratios)

        # Obtain the chroma features with the function of the librosa library
        chroma = lbs.feature.chroma_stft(y=audio_data, sr=sr, n_fft=flen, hop_length=hop)
        # Obtain the mean and standard deviation
        M[i, 83] = np.nanmean(chroma)
        M[i, 84] = np.std(chroma)

        # Obtain the spectral flux from a function generated in the audio features script
        spectral_flux = af.get_spectral_flux(audio_data, flen=flen, hop=hop)
        # Obtain the mean and standard deviation
        M[i, 85] = np.nanmean(spectral_flux)
        M[i, 86] = np.std(spectral_flux)

        if verbose:
            print('Done')
    return M

##############################################################################
# Data read (and prepare)
##############################################################################

major_files = listdir('./data/Major')
minor_files = listdir('./data/Minor')
major_files = ['./data/Major/' + f for f in major_files]
minor_files = ['./data/Minor/' + f for f in minor_files]

X = deepcopy(minor_files)
X.extend(major_files)
y = list(np.concatenate((np.zeros(len(minor_files)),
                         np.ones(len(major_files))), axis=0).astype(int))

test_size = 0.3
ran_seed = 999
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=test_size,
                                                    random_state=ran_seed)

##############################################################################
# Feature Correlation Analysis
##############################################################################

# As we have many variables correlation is likely to appear

# The feature names are collected to then show which variables are being taking out
feature_names = [
    'Energy Entropy Mean', 'Energy Entropy Max',
    *['MFCC Mean {}'.format(i) for i in range(1, 14)],
    *['MFCC Std {}'.format(i) for i in range(1, 14)],
    *['Delta MFCC Mean {}'.format(i) for i in range(1, 14)],
    *['Delta-Delta MFCC Mean {}'.format(i) for i in range(1, 14)],
    *['Skewness MFCC {}'.format(i) for i in range(1, 14)],
    *['Kurtosis MFCC {}'.format(i) for i in range(1, 14)],
    'Spectral Entropy Mean', 'Spectral Centroid Mean',
    'Harmonic Ratio Mean', 'Chroma Mean', 'Chroma Std',
    'Spectral Flux Mean', 'Spectral Flux Std'
]

# The correlation matrix was computed in other iteration, so it is loaded from CSV
correlation_matrix = pd.read_csv('correlation_matrix.csv', index_col=0)

# Identify and drop highly correlated features which surpass 0.8 in the correlation matrix
correlation_threshold = 0.8
to_drop = set()

# Check all relationships to see which ones surpass the threshold
for i in range(correlation_matrix.shape[0]):
    for j in range(i + 1, correlation_matrix.shape[1]):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            # Take only one of the variables to not drop both of them
            to_drop.add(correlation_matrix.columns[j])

print("Features to drop:", to_drop)

##############################################################################
# Feature Reduction
##############################################################################

M_train = extract_features(X_train)
df_train = pd.DataFrame(M_train, columns=feature_names)
# Drop the columns we have kept in the variable in both the training and test set
M_train_reduced = df_train.drop(columns=to_drop)

M_test = extract_features(X_test)
df_test = pd.DataFrame(M_test, columns=feature_names)
M_test_reduced = df_test.drop(columns=to_drop)

##############################################################################
# Model Training and Evaluation
##############################################################################

# Scale the variables
scaler = StandardScaler().fit(M_train_reduced)
M_train_n = scaler.transform(M_train_reduced)

# Use a support vector classifier with tuned parameters
clf = SVC(C=1, gamma=1, kernel='rbf', probability=True)
# Fit the model with the data
clf.fit(M_train_n, y_train)

# Test the model with the test, applying the same process as the training
M_test_n = scaler.transform(M_test_reduced)
# Predict the probabilities and compute the ROC curve
y_scores = clf.predict_proba(M_test_n)[:, 1]
false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_scores)
auc_svm = roc_auc_score(y_test, y_scores)

# Plot it
plt.subplots(1, figsize=(10, 10))
plt.title(f'ROC Curve - AUC = {np.round(auc_svm, 3)}')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
