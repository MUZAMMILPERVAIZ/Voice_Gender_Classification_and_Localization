import cv2
import pyaudio
import wave
from tqdm import tqdm
from time import sleep
import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import librosa.display
from colored import fg, attr

model = tf.keras.models.load_model("gender_classifier_v2.h5")

green = fg('green')
red = fg('red')
reset_color = attr('reset')


def pre_gen_binarize(img):
    img = img[:, :, 0]
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    binary_img = binary_img.reshape(binary_img.shape[0], binary_img.shape[1], 1)
    return binary_img // 255


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()
print("[Info] Kindly Speak for 5 seconds...")

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("[Info] Recording...")

frames = []

for i in tqdm(range(0, int(RATE / CHUNK * RECORD_SECONDS)), colour="green"):
    data = stream.read(CHUNK)
    frames.append(data)

print("[Info] done recording...")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print("[Info] Creating Spectogram...")

# Randomly choosing one of the audio files
filename = "output.wav"
# Checking how the image looks like with the frequency restriction
X, sample_rate = librosa.load(filename, sr=None, res_type='kaiser_fast')
# Setting the size of the image
fig = plt.figure(figsize=[1, 1])
# This is to get rid of the axes and only get the picture
ax = fig.add_subplot(111)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.set_frame_on(False)
S = librosa.feature.melspectrogram(y=X, sr=sample_rate)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', fmin=50, fmax=280)
plt.savefig("output.jpg", dpi=500, bbox_inches='tight', pad_inches=0)

print("[Info] Reading Spectogram as image...")
img = cv2.resize(cv2.imread("output.jpg"), (64, 64))

print("[Info] Preprocessing image...")
image = pre_gen_binarize(img)
image_input = np.array([image], )

print("[Info] Loading Tensorflow model...")

print("[Info] Predicting voice from model...")
pred = model.predict(image_input)

if (np.argmax(pred) == 1):
    print("[Result] The gender of speaker is Male.....")
    if ((pred[0][1] > 0.6) and (pred[0][2] < 0.4)):
        print("["+ green + "Distance" + reset_color+ "] The Speaker is inside the room.....")
    else:
        print("["+ red + "Distance" + reset_color+ "] The Speaker is far from Mic possibly outside the room.....")

elif (np.argmax(pred) == 0):
    print("[Result] The gender of speaker is Female.....")
    if ((pred[0][0] > 0.6) and (pred[0][2] < 0.4)):
        print("["+ green + "Distance" + reset_color+ "] The Speaker is inside the room.....")
    else:
        print("["+ red + "Distance" + reset_color+ "] The Speaker is far from Mic possibly outside the room.....")

elif (np.argmax(pred) == 2):
    print("[Result] Too much noise Speaker is not audible.....")
