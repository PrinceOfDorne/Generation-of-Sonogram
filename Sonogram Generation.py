#PART 1 - RECORDING AUDIO

import pyaudio
import wave
from array import array

FORMAT=pyaudio.paInt16
CHANNELS=2
RATE=44100
CHUNK=1024
RECORD_SECONDS=15
FILE_NAME="RECORDING.wav"

audio=pyaudio.PyAudio() #instantiate the pyaudio

#recording prerequisites
stream=audio.open(format=FORMAT,channels=CHANNELS, 
                  rate=RATE,
                  input=True,
                  frames_per_buffer=CHUNK)

#starting recording
frames=[]

for i in range(0,int(RATE/CHUNK*RECORD_SECONDS)):
    data=stream.read(CHUNK)
    data_chunk=array('h',data)
    vol=max(data_chunk)
    if(vol>=500):
        print("something said")
        frames.append(data)
    else:
        print("nothing")
    print(i)
    print("\n")


#end of recording
stream.stop_stream()
stream.close()
audio.terminate()
#writing to file
wavfile=wave.open(FILE_NAME,'wb')
wavfile.setnchannels(CHANNELS)
wavfile.setsampwidth(audio.get_sample_size(FORMAT))
wavfile.setframerate(RATE)
wavfile.writeframes(b''.join(frames))#append frames recorded to file
wavfile.close()


#2 - Creating Sonogram

import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.io import wavfile
from scipy.fftpack import fft

samplingFreq, mySound = wavfile.read(FILE_NAME)

float(mySound.shape[1])

mySoundDataType = mySound.dtype

mySound = mySound / (2.**15)

mySoundShape = mySound.shape

samplePoints = float(mySound.shape[0])

signalDuration =  mySound.shape[0] / samplingFreq

mySoundOneChannel = mySound[:,0]

timeArray = np.arange(0, samplePoints, 1)
timeArray = timeArray / samplingFreq
timeArray = timeArray * 1000

plt.plot(timeArray, mySoundOneChannel, color='G')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.savefig("T vs. A.jpg")

mySoundLength = len(mySound)

fftArray = fft(mySoundOneChannel)

numUniquePoints = np.ceil((mySoundLength + 1) / 2.0)
fftArray = fftArray[0:int(numUniquePoints)]
fftArray = abs(fftArray)
fftArray = fftArray / float(mySoundLength)
fftArray = fftArray **2

if mySoundLength % 2 > 0: #we've got odd number of points in fft
    fftArray[1:len(fftArray)] = fftArray[1:len(fftArray)] * 2
else: #We've got even number of points in fft
    fftArray[1:len(fftArray) -1] = fftArray[1:len(fftArray) -1] * 2

freqArray = np.arange(0, numUniquePoints, 1.0) * (samplingFreq / mySoundLength);

plt.plot(freqArray/1000, 10 * np.log10 (fftArray), color='B')
plt.xlabel('Frequency (Khz)')
plt.ylabel('Power (dB)')
plt.savefig('F vs. P.png')
plt.show()
