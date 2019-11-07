#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import librosa
import numpy as np
from magenta.common import tf_utils
from magenta.music import audio_io
import magenta.music as mm
from magenta.models.onsets_frames_transcription import configs
from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription import data
from magenta.models.onsets_frames_transcription import split_audio_and_label_data
from magenta.models.onsets_frames_transcription import train_util
from magenta.music import midi_io
from magenta.protobuf import music_pb2
from magenta.music import sequences_lib

## Define model and load checkpoint
## Only needs to be run once.
CHECKPOINT_DIR = '/Users/bigwhite/Documents/BIGCHUANG/train_checkpoint'
config = configs.CONFIG_MAP['onsets_frames']
hparams = config.hparams
hparams.use_cudnn = False
hparams.batch_size = 1

examples = tf.placeholder(tf.string, [None])

dataset = data.provide_batch(
    examples=examples,
    preprocess_examples=True,
    hparams=hparams,
    is_training=False)

estimator = train_util.create_estimator(
    config.model_fn, CHECKPOINT_DIR, hparams)

iterator = dataset.make_initializable_iterator()
next_record = iterator.get_next()


# In[13]:



import pickle



import wave
def read_wave_data(file_path):
	#open a wave file, and return a Wave_read object
	f = wave.open(file_path,"rb")
	#read the wave's format infomation,and return a tuple
	params = f.getparams()
	#get the info
	nchannels, sampwidth, framerate, nframes = params[:4]
	#Reads and returns nframes of audio, as a string of bytes. 
	str_data = f.readframes(nframes)
	#close the stream
	f.close()
	#turn the wave's data to array
	wave_data = np.fromstring(str_data, dtype = np.short)
	#for the data is stereo,and format is LRLRLR...
	#shape the array to n*2(-1 means fit the y coordinate)
# 	wave_data.shape = -1, 2
# 	#transpose the data
# 	wave_data = wave_data.T
# 	#calculate the time bar
# 	time = np.arange(0, nframes) * (1.0/framerate)
	return wave_data

name = '/Users/bigwhite/Documents/BIGCHUANG/test.wav'
wave_data = read_wave_data(name) 
uploaded = {name:wave_data}
to_process = []
print(wave_data.shape)
# for fn in uploaded.keys():
#   print('User uploaded file "{name}" with length {length} bytes'.format(
#       name=fn, length=len(uploaded[fn])))
#   wav_data = uploaded[fn]
wav_data = wave_data
example_list = list(
      split_audio_and_label_data.process_record(
          wav_data=wav_data,
          ns=music_pb2.NoteSequence(),
          example_id=fn,
          min_length=0,
          max_length=-1,
          allow_empty_notesequence=True))
assert len(example_list) == 1
to_process.append(example_list[0].SerializeToString())
  
# print('Processing complete for', fn)
  
sess = tf.Session()

sess.run([
    tf.initializers.global_variables(),
    tf.initializers.local_variables()
])

sess.run(iterator.initializer, {examples: to_process})

def input_fn(params):
  del params
  return tf.data.Dataset.from_tensors(sess.run(next_record))


# In[ ]:


prediction_list = list(
    estimator.predict(
        input_fn,
        yield_single_examples=False))
assert len(prediction_list) == 1

frame_predictions = prediction_list[0]['frame_probs_flat'] > .5
onset_predictions = prediction_list[0]['onset_probs_flat'] > .5
velocity_values = prediction_list[0]['velocity_values_flat']

sequence_prediction = sequences_lib.pianoroll_to_note_sequence(
    frame_predictions,
    frames_per_second=data.hparams_frames_per_second(hparams),
    min_duration_ms=0,
    min_midi_pitch=constants.MIN_MIDI_PITCH,
    onset_predictions=onset_predictions,
    velocity_values=velocity_values)

# Ignore warnings caused by pyfluidsynth
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

mm.plot_sequence(sequence_prediction)
mm.play_sequence(sequence_prediction, mm.midi_synth.fluidsynth,
                 colab_ephemeral=False)


# In[ ]:


midi_filename = ('prediction.mid')
midi_io.sequence_proto_to_midi_file(sequence_prediction, midi_filename)

files.download(midi_filename)

