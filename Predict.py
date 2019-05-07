from Utils import *
from classifier.CNNClassifier import CNNClassifier
from classifier.Classifier import Classifier
from classifier.ConstantClassifier import ConstantClassifier
from classifier.LinearClassifier import LinearClassifier
from classifier.RFClassifier import RFClassifier
from classifier.SNNClassifier import SNNClassifier
from Settings import *
from sklearn.model_selection import train_test_split
from main import run_for_classifier
from auditok import ADSFactory, AudioEnergyValidator, StreamTokenizer, player_for, dataset

SAVE=True
LOAD=True
#classifier = CNNClassifier(verbose=1)
#classifier = SNNClassifier(num_units=64, verbose=1)
classifier = RFClassifier(n_estimators=100)
pathGender = MODELS_DIR + classifier.get_classifier_name() + DUMP_EXT
print(pathGender)
if os.path.isfile(pathGender):
    print("Begin to Load model")
    classifier.load(pathGender)
    #model.save_weights("gender_cnn_model_weight.h5")
    #json_string = model.to_json()
    #open('gender_cnn_model_json.json','w').write(json_string)
    #model.save("gender_cnn_model.h5")
else:
    print("Model file not exist.{}".format(path))

def f(path):
    #if path.endswith(AUDIO_EXT) == False and path.endswith(AUDIO_EXT_FLAC) == False and path.endswith(AUDIO_EXT_MP3) == False and path.endswith(AUDIO_EXT_PCM) == False:
    if path.endswith(AUDIO_EXT_PCM) == False:
        return False
    return True

def list_files(dir_name: str, ext=AUDIO_EXT) -> np.ndarray:
    """
    List the files in a directory recursively for a given extension.
    :param dir_name: The directory to search
    :param ext: The extension of the files to search for
    :return: The array of filenames
    """
    return np.asarray(list(map(lambda path: path.replace("\\", PATH_SEPARATOR),
                               filter(lambda path: f(path),
                                      [os.path.join(dp, f) for dp, dn, fn in os.walk(dir_name) for f in fn]))))

def convertPCM(destPath, srcPath):
    #print(srcPath)
    outPath = destPath
    fout = open(outPath,'wb') #用二进制的写入模式
    #fout.write(struct.pack('4s','\x66\x6D\x74\x20'))
    #写入一个长度为4的串，这个串的二进制内容为 66 6D 74 20
    #Riff_flag,afd,fad,afdd, = struct.unpack('4c',fin.read(4))
    #读入四个字节，每一个都解析成一个字母
    #open(sys.argv[4],'wb').write(struct.pack('4s','fmt '))
    #将字符串解析成二进制后再写入
    #open(sys.argv[4],'wb').write('\x3C\x9C\x00\x00\x57')
    #直接写入二进制内容：3C 9C 00 00 57
    #fout.write(struct.pack('i',6000)) #写入6000的二进制形式
    #check whether inFile has head-Info
    inPath= srcPath
    fin = open(inPath,'rb')
    Riff_flag, = struct.unpack('4s',fin.read(4))
    if Riff_flag == 'RIFF':
        #print("%s have head" % inPath)
        fin.close()
        #sys.exit(0)
    else:
        #print("%s no head" % inPath)
        fin.close()
        #采样率
        sampleRate = int(16000)
        #bit位
        bits = int(16)
        fin = open(inPath,'rb')
        startPos = fin.tell()
        fin.seek(0,os.SEEK_END)
        endPos = fin.tell()
        sampleNum = int(endPos - startPos)
        #print(sampleNum)
        #headInfo = geneHeadInfo(sampleRate,bits,sampleNum)
        #fout.write(headInfo)
        fout.write('\x52\x49\x46\x46'.encode())
        fout.write(struct.pack('i',sampleNum + 36))
        #fout.write(fileLength)
        fout.write('\x57\x41\x56\x45\x66\x6D\x74\x20\x10\x00\x00\x00\x01\x00\x01\x00'.encode())
        fout.write(struct.pack('i',sampleRate))
        fout.write(struct.pack('i',int(sampleRate * bits / 8)))
        fout.write('\x02\x00'.encode())
        fout.write(struct.pack('H',bits))
        fout.write('\x64\x61\x74\x61'.encode())
        fout.write(struct.pack('i',sampleNum))
        fin.seek(os.SEEK_SET)
        fout.write(fin.read())
        fin.close()
        fout.close()
        # We set the `record` argument to True so that we can rewind the source
        asource = ADSFactory.ads(filename=destPath, record=True)
        validator = AudioEnergyValidator(sample_width=asource.get_sample_width(), energy_threshold=50)
        # Default analysis window is 10 ms (float(asource.get_block_size()) / asource.get_sampling_rate())
        # min_length=20 : minimum length of a valid audio activity is 20 * 10 == 200 ms
        # max_length=400 :  maximum length of a valid audio activity is 400 * 10 == 4000 ms == 4 seconds
        # max_continuous_silence=30 : maximum length of a tolerated  silence within a valid audio activity is 30 * 30 == 300 ms
        tokenizer = StreamTokenizer(validator=validator, min_length=50, max_length=400, max_continuous_silence=1)
        asource.open()
        tokens = tokenizer.tokenize(asource)
        # Play detected regions back
        #player = player_for(asource)
        #print("\n ** playing detected regions...\n")
        data = b''

        for i,t in enumerate(tokens):
            #print("Token [{0}] starts at {1} and ends at {2}".format(i+1, t[1], t[2]))
            data = data + b''.join(t[0])
        #player.play(data)

        sampleNum = len(data)
        if sampleNum>1000:
            #采样率
            sampleRate = int(16000)
            #bit位
            bits = int(16)
            fout = open(srcPath+".wav",'wb')
            fout.write('\x52\x49\x46\x46'.encode())
            fout.write(struct.pack('i',sampleNum + 36))
            #fout.write(fileLength)
            fout.write('\x57\x41\x56\x45\x66\x6D\x74\x20\x10\x00\x00\x00\x01\x00\x01\x00'.encode())
            fout.write(struct.pack('i',sampleRate))
            fout.write(struct.pack('i',int(sampleRate * bits / 8)))
            fout.write('\x02\x00'.encode())
            fout.write(struct.pack('H',bits))
            fout.write('\x64\x61\x74\x61'.encode())
            fout.write(struct.pack('i',sampleNum))
            fout.write(data)
            fout.close()
            #assert len(tokens) == 8
            asource.close()

def read_audio(path, target_fs=None):
    #print(path)
    if path.upper().endswith(".PCM"):
        convertPCM(TEMP_WAV, path)
        path = path+".wav"
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs

def audio_to_features(filename: str, n_features: int = FEATURES_NUMBER) -> np.ndarray:
    """
    Extract MFCC features from audio file using librosa.
    :param filename: The name of the file
    :param n_features: The number of features to extract
    :return: An ndarray of features
    """
    #data, samplerate = librosa.load(filename, sr=None)
    fs = cfg.sample_rate
    print(filename)
    data, samplerate = read_audio(filename, target_fs=fs)
    mfcc_features = np.asarray(librosa.feature.mfcc(data, samplerate, n_mfcc=n_features))
    global min_shape
    if mfcc_features.shape[1] < min_shape:  # Keep track of min_shape for 2D input
        min_shape = mfcc_features.shape[1]
    return mfcc_features.transpose()

def pre_file_to_features_with_labels(filename: str) -> Any:
    """
    Extract features and label from an audio file.
    :param filename: The filename
    :return: A tuple (features, label)
    """
    features = audio_to_features(filename)
    #print(features)
    return features, 0

def pre_files_to_features(filenames: Iterable[str]) -> np.ndarray:
    """
    Extract features and labels from a list of files.
    :param filenames: The filenames to use
    :return: An array of (features, label) tuples
    """
    features_with_label = np.asarray([pre_file_to_features_with_labels(file) for file in filenames])
    flattened_features = flatten(extract_features(features_with_label))
    #print(flattened_features.min(axis=0))
    #print(flattened_features.max(axis=0))
    flattened_features_min_f = flattened_features.min(axis=0)
    flattened_features_max_f = flattened_features.max(axis=0)
    if not os.path.isfile(MIN_FEATURES_FILE) or not os.path.isfile(MAX_FEATURES_FILE):
        min_f = flattened_features.min(axis=0)
        #save_nparray(min_f, MIN_FEATURES_FILE)
        max_f = flattened_features.max(axis=0)
        #save_nparray(max_f, MAX_FEATURES_FILE)
    else:
        min_f = load_nparray(MIN_FEATURES_FILE)
        max_f = load_nparray(MAX_FEATURES_FILE)
        #print(min_f)
        #print(max_f)
    min_f = np.asarray([min(min_f[i] , flattened_features_min_f[i]) for i in range(len(min_f))])
    max_f = np.asarray([max(max_f[i] , flattened_features_max_f[i]) for i in range(len(max_f))])
    save_nparray(min_f, MIN_FEATURES_FILE)
    save_nparray(max_f, MAX_FEATURES_FILE)
    #print(min_f)
    #print(max_f)
    #print("end")
    # Normalize the features
    features_with_label = np.asarray(list(
        map(lambda feat_label_tuple: (
            np.asarray(list(map(lambda sample: (sample - min_f) / (max_f - min_f), feat_label_tuple[0]))),
            feat_label_tuple[1]),
            features_with_label)))
    #print(features_with_label)
    #print("feature")
    return features_with_label

def pre_PredictGender(path):
    #print(path)
    print("Begin to load audio & extract features")
    pre_features_with_label = pre_files_to_features(list_files(path))
    #pre_set = to_2d(pre_features_with_label)
    pre_set = to_1d(pre_features_with_label)
    pre_features = extract_features(pre_set)
    pre_label = extract_labels(pre_set)
    print("Begin to predict")
    #results_pre = modelGender.predict(pre_features, batch_size=64, verbose=1)
    results_pre = classifier.predict(pre_features)
    #print(results_pre)
    res = clamp(results_pre)
    print(res)
    print("male:{}".format(np.sum(res)))
    print("female:{}".format(len(res)-np.sum(res)))
    return return_majority(res)
    #transformed_test_set = to_1d(pre_features_with_label)
    #samples_features = extract_features(transformed_test_set)
    #samples_predictions = classifier.predict(samples_features)
    #print(samples_predictions)
    #print(samples_predictions[0])

def predictGender(path):
    try:
        value = pre_PredictGender(path)
        result = "Unknown"
        if value == 1:
            result = "Male"
        elif value == 0:
            result = "Female"
        print(result)
        return result
    except Exception as e:
        print("Error occur, Unknown")
        return "Unknown"

def main():
    return predictGender("predict_dir/Random_A5")

if __name__ == "__main__":
    main()
