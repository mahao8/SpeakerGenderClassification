#coding=utf-8
import requests
url = "http://109.123.100.102:8080"
path = "/Users/yancx/SC/tensorflow3/SpeakerAgeClassificationTimit/predict_dir/Random_A1/MAHH0/mahao_VrUsbPcm_003_vc.pcm"
#path = "/Users/yancx/SC/tensorflow3/backup/SpeakerGenderClassificationTimit_0.3.okay/predict_dir/Test/FASW0/SA1.WAV"
print (path)
files = {'file': open(path, 'rb')}
print(files)
r = requests.post(url, files=files)
print (r.url)
print (r.text)
