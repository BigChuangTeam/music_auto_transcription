import numpy, scipy, matplotlib.pyplot as plt, IPython.display as ipd
import librosa, librosa.display
import stanford_mir; stanford_mir.init()
sr = 22050
def generate_tone(midi):
    T = 0.5
    t = numpy.linspace(0, T, int(T*sr), endpoint=False)
    f = librosa.midi_to_hz(midi)
    return numpy.sin(2*numpy.pi*f*t)
x = numpy.concatenate([generate_tone(midi) for midi in [48, 52, 55, 60, 64, 67, 72, 76, 79, 84]])
spectral_novelty = librosa.onset.onset_strength(x, sr=sr)
frames = numpy.arange(len(spectral_novelty))
t = librosa.frames_to_time(frames, sr=sr)
plt.figure(figsize=(15, 4))
plt.plot(t, spectral_novelty, 'r-')
plt.xlim(0, t.max())
plt.xlabel('Time (sec)')
plt.legend(('Spectral Novelty',))
hop_length = 512
tempogram = librosa.feature.tempogram(onset_envelope=spectral_novelty, sr=sr,hop_length=hop_length)
ac_global = librosa.autocorrelate(spectral_novelty, max_size=tempogram.shape[0])
ac_global = librosa.util.normalize(ac_global)
tempo = librosa.beat.tempo(onset_envelope=spectral_novelty, sr=sr, hop_length=hop_length)[0]
plt.figure(figsize=(15, 3))
plt.plot(4, 1, 2)
librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='tempo')
plt.axhline(tempo, color='w', linestyle='--', alpha=1, label='Estimated tempo={:g}'.format(tempo))
plt.legend(frameon=True, framealpha=0.75)