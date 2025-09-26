import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt



# .wav 파일 경로 설정
wav_folder_path = "C:/Users/11e26/Desktop/internship/source/clear_command"


def set_audio(path):
    freq, audio = wavfile.read(path) 
    audio = audio.astype(float) / np.max(np.abs(audio))  
    return freq, audio


def rate_coding(audio, n_neurons=100, dt=1e-3):
    n_time_steps = len(audio)
    spike_train = np.zeros((n_neurons, n_time_steps))
    for t in range(n_time_steps):
        p = np.clip(np.abs(audio[t]), 0, 1)  
        spike_train[:, t] = np.random.rand(n_neurons) < p
    return spike_train, n_time_steps


if __name__=="__main__":
    for i in range(1, 20):
        freq, audio = set_audio(wav_folder_path + "/alexa_sample"+str(i)+".wav")
        print(f"Sample freq: {freq}, Audio Length: {len(audio)}, {i}th file")
        spike_train, n_time_steps = rate_coding(audio)
        print(spike_train.shape)  
        plt.figure(figsize=(10, 4))
        plt.imshow(spike_train[:, :n_time_steps], cmap='binary', aspect='auto')
        plt.xlabel("Time step")
        plt.ylabel("Neuron index")
        plt.title(f"Spike Train {i}th file")
        plt.savefig(f"spike_train_{i}.png")