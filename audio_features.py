# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 19:29:03 2021

@author: epla
"""

import numpy as np
import librosa as lbs
from scipy.fft import fft
from librosa.feature import melspectrogram
from scipy.signal.windows import hamming

def preprocess_audio(x, thr=60):
  """
  Perform preprocessing of an audio signal (Trimming and centering)
    ARGUMENTS:
      x: audio data
      thr: threshold below reference to consider as silence (dB) 
  """
  y, _ = lbs.effects.trim(x, top_db=thr)
  y = y - np.mean(y)
  return y

##############################################################################
# Descritor functions
##############################################################################
def get_energy(x, flen=1024, hop=256):
  """
  Computes energy of an audio signal (use hamming window)
    ARGUMENTS:
      x: audio data
      flen: frame length
      hop: Hop length
  """
  # Split the signal into frames
  frames = lbs.util.frame(x, frame_length=flen, hop_length=hop)
  
  # Window the frames with hamming window
  frames = frames * hamming(flen)[...,np.newaxis]
  
  # Get frame energies
  energies = np.sum(frames**2, axis=0)
  
  return energies


def get_energy_entropy(x, flen=1024, hop=256, nsub=10):
  """
  Computes energy entropy of an audio signal
    ARGUMENTS:
      x: audio data
      flen: frame length
      hop: Hop length
      nsub: Number of subframes
  """
  # Get sub-frame length
  subf_len = flen // nsub
  
  # Split the signal into frames
  frames = lbs.util.frame(x, frame_length=flen, hop_length=hop)
  
  #Get entropy for each frame
  eps = np.finfo(np.float64).eps
  nframes = frames.shape[1]
  entropies = np.zeros(nframes)
  for i in range(nframes):
    ienergies = get_energy(frames[:,i], flen=subf_len, hop=subf_len)
    ienergies = ienergies / (np.sum(ienergies) + eps) # normalization
    entropies[i] = -np.sum(ienergies * np.log2(ienergies + eps)) # entropy
  
  return entropies


def get_spectral_entropy(x, flen=1024, hop=256, nsub=8):
  """
  Computes spectral entropy of an audio signal
    ARGUMENTS:
      x: audio data
      flen: frame length
      hop: Hop length
      nsub: Number of subframes
  """
  # Split the signal into frames
  frames = lbs.util.frame(x, frame_length=flen, hop_length=hop)
  
  # Window the frames
  frames = frames * hamming(flen)[...,np.newaxis]
  
  # Positive spectrum length
  nspec = int(flen // 2)
  
  # Sub-frame length
  subf_len = int(np.floor(nspec / nsub))
  
  #Get entropy for each frame
  eps = np.finfo(np.float64).eps
  nframes = frames.shape[1]
  entropies = np.zeros(nframes)
  for i in range(nframes):
    ith_pow_spec = np.abs(fft(frames[:,i])[0:nspec]) ** 2
    
    # If nsub is not a divisor of nsec, remove exceeding spectrum samples
    if nspec > subf_len * nsub:
        ith_pow_spec = ith_pow_spec[0:subf_len * nsub]

    # Define sub-frames (using matrix reshape)
    ith_subframes = ith_pow_spec.reshape(subf_len, nsub, order='F').copy()

    # Compute normalized spectral sub-energies
    subf_nenergies = np.sum(ith_subframes, axis=0) / (np.sum(ith_pow_spec) + eps)
    
    entropies[i] = -np.sum(subf_nenergies * np.log2(subf_nenergies + eps)) # entropy
  
  return entropies


def get_spectral_centroid(x, sr, flen=1024, hop=256):
  """
  Computes spectral centroid of an audio signal
    ARGUMENTS:
      x: audio data
      sr: sampling rate
      flen: frame length
      hop: Hop length
  """
  spcentroids = lbs.feature.spectral_centroid(y=x, sr=sr, n_fft=flen, hop_length=hop)
  spcentroids = np.squeeze(spcentroids)
  return spcentroids


def get_spectral_spread(x, sr, flen=1024, hop=256, porder=2):
  """
  Computes spectral spread of an audio signal
    ARGUMENTS:
      x: audio data
      sr: sampling rate
      flen: frame length
      hop: Hop length
      porder: Power to raise deviation from spectral centroid
  """
  spread = lbs.feature.spectral_bandwidth(y=x, sr=sr, n_fft=flen, hop_length=hop, p=porder)
  spread = np.squeeze(spread)
  return spread


def get_zero_crossing_rate(x, flen=1024, hop=256):
  """
  Computes the zero-crossing rate of an audio signal
    ARGUMENTS:
      x: audio data
      flen: frame length
      hop: Hop length
  """
  zcr = lbs.feature.zero_crossing_rate(x, frame_length=flen, hop_length=hop)
  zcr = np.squeeze(zcr)
  return zcr


def harmonic(frame, sr):
  """
  Computes harmonic ratio and pitch
    ARGUMENTS:
      frame: audio data frame
      sr: sampling rate
  """
  m = int(np.round(0.016 * sr)) - 1
  r = np.correlate(frame, frame, mode='full')

  g = r[len(frame) - 1]
  r = r[len(frame):-1]

  # estimate m0 (as the first zero crossing of R)
  [a, ] = np.nonzero(np.diff(np.sign(r)))

  if len(a) == 0:
    m0 = len(r) - 1
  else:
    m0 = a[0]
  if m > len(r):
    m = len(r) - 1

  gamma = np.zeros(int(m), dtype=np.float64)
  eps = np.finfo(np.float64).eps
  
  cumulative_sum = np.cumsum(frame ** 2)
  gamma[m0:m] = r[m0:m] / (np.sqrt((g * cumulative_sum[m:m0:-1])) + eps)

  zcr = lbs.feature.zero_crossing_rate(gamma, frame_length=gamma.size, center=False)

  if zcr > 0.15:
    hr = 0.0
    f0 = 0.0
  else:
    if len(gamma) == 0:
      hr = 1.0
      blag = 0.0
      gamma = np.zeros((m), dtype=np.float64)
    else:
      hr = np.max(gamma)
      blag = np.argmax(gamma)

    # Get fundamental frequency:
    f0 = sr / (blag + eps)
    if f0 > 5000:
      f0 = 0.0
    if hr < 0.1:
      f0 = 0.0

  return hr, f0


def get_harmonic_ratio(x, sr, flen=1024, hop=256):
  """
  Computes harmonic ratio of an audio signal
    ARGUMENTS:
      x: audio data
      sr: sampling rate
      flen: frame length
      hop: Hop length
  """
  # Split the signal into frames
  frames = lbs.util.frame(x, frame_length=flen, hop_length=hop)
  
  # Compute the hr for each frame
  hr_frames = np.zeros(frames.shape[1])
  for i in range(hr_frames.size):
    hr_frames[i],_ = harmonic(frames[:,i], sr)
  return hr_frames


def get_mfccs(x, sr, flen=1024, hop=256, n_mfcc=20):
  """
  Computes the Mel-frequency cepstral coefficients
  ARGUMENTS:
      x: audio data
      sr: sampling rate
      flen: frame length
      hop: Hop length
      n_mfcc: number of coefficients to return
  """
  S = lbs.feature.melspectrogram(y=x, sr=sr, n_fft=flen, hop_length=hop)
  mfcc_frames = lbs.feature.mfcc(sr=sr, S=S, n_mfcc=n_mfcc)
  return mfcc_frames.T


def get_spectral_flux(x, flen=1024, hop=256):
  """
  Computes the spectral flux of an audio signal.
  ARGUMENTS:
    x: audio data
    flen: frame length
    hop: Hop length
  """

  # Firstly compute the Short-Time Fourier Transform with the librosa function
  stft = lbs.stft(x, n_fft=flen, hop_length=hop)
  # Take the magnitude with the absolute value
  magnitude_spectra = np.abs(stft)
  # Apply the flux formula (difference in magnitude between consecutive time frames for each frequency bin)
  flux = np.sum(np.diff(magnitude_spectra, axis=1) ** 2, axis=0)
  return flux

