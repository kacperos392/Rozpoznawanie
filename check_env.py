import sys

print("Python:", sys.version)

import numpy as np
import pandas as pd
import matplotlib
import librosa
import soundfile as sf
import sklearn
import tensorflow as tf
import streamlit

print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("matplotlib:", matplotlib.__version__)
print("librosa:", librosa.__version__)
print("soundfile:", sf.__version__)
print("scikit-learn:", sklearn.__version__)
print("tensorflow:", tf.__version__)
print("streamlit:", streamlit.__version__)

# Mały test: czy Librosa potrafi policzyć spektrogram Mel
y = np.random.randn(22050).astype("float32")  # 1 s losowego szumu
mel = librosa.feature.melspectrogram(y=y, sr=22050, n_mels=128)
print("Kształt spektrogramu Mel:", mel.shape)

# Mały test: czy TensorFlow działa
print("Test TF:", tf.reduce_sum(tf.constant([1.0, 2.0, 3.0])).numpy())
print("GPU:", tf.config.list_physical_devices("GPU"))