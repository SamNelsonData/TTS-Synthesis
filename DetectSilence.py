
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS

import numpy as np

import librosa

import matplotlib.pyplot as plt
import librosa.display


path = './testAudio.wav'

'''
[Fs, x] = aIO.read_audio_file(path)
segments = aS.silence_removal(x, 
							  Fs, 
							  .02, 
							  .02, 
							  weight=.3,
							  plot=True)
'''

x,sr = librosa.load(path, sr=44000)

plt.figure(figsize=(14,5))

librosa.display.waveshow(x, sr = sr)

hop_length = 256
frame_length = 512
energy = np.array([
			sum(abs(x[i:i+frame_length]**2))
			for i in range(0, len(x), hop_length)
])

print(len(x))
print(len(energy))

plt.show()
