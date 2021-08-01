import re
import os
from datasets.arrow_dataset import DatasetTransformationNotAllowedError
import torch
import librosa
import warnings
import pandas as pd

from datasets import load_dataset, load_metric, Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoTokenizer, AutoModel

# Specify which
#     - Datasets
#     - Models both English and German
#     - Is the test run local or in the cluster

dataset = 'torgo'
models_en = ['jonatasgrosman/wav2vec2-large-xlsr-53-english']# ['facebook/wav2vec2-base-960h', 'facebook/s2t-small-librispeech-asr', 'facebook/wav2vec2-large-960h-lv60-self']
l = False # True if run local, False if cluster


def run_test(dataset, model, language, local):
    LANG_ID = language
    MODEL_ID = model
    DEVICE = 'cpu'
    if not local:
        DEVICE = 'cuda'

    wer = load_metric('wer.py')
    cer = load_metric('cer.py')

    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.to(DEVICE)

    #---------------------------------------------------------------------


    def speech_file_to_array(x):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            speech_array, sampling_rate = librosa.load(x, sr=16_000)
        return speech_array


    def evaluate(batch):
        inputs = processor(batch['speech'], sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = model(inputs.input_values.to(DEVICE), attention_mask=inputs.attention_mask.to(DEVICE)).logits

        pred_ids = torch.argmax(logits, dim=-1)
        batch['pred_text'] = processor.batch_decode(pred_ids)
        return batch


    def get_predicted_sentence(file):
        input = processor(file, sampling_rate=16000,
                        return_tensors="pt", padding=True)

        with torch.no_grad():
            logit = model(input.input_values,
                        attention_mask=input.attention_mask).logits

        predicted_id = torch.argmax(logit, dim=-1)
        predicted_sentence = processor.batch_decode(predicted_id)

        return predicted_sentence


    #---------------------------------------------------------------------

    # Method to import Torgo datasets
    def import_torgo(location):
        print('Import Torgo dataset')
        speakers = os.listdir(location)
        dfs = []

        for speaker in speakers:
            df = pd.DataFrame(columns=['id', 'file', 'sentence', 'severity'])
            l = os.path.join(location, speaker)
            if os.path.isdir(l):
                sessions = os.listdir(l)
                for session in sessions:
                    if session[0:7] == 'Session':
                        s = os.path.join(l, session)

                        if os.path.isdir(os.path.join(s, 'wav_arrayMic')):
                            recordings_location = os.path.join(s, 'wav_arrayMic')
                        else:
                            recordings_location = os.path.join(s, 'wav_headMic')

                        recordings = os.listdir(recordings_location)
                        for recording in recordings:
                            if len(recording) == 8:
                                if os.path.isfile(os.path.join(s, 'prompts', (recording[:-4] + '.txt'))):
                                    sentence = open(os.path.join(
                                        s, 'prompts', (recording[:-4] + '.txt')))
                                    new_row = {'id': speaker + '_' + session + '_' + recording[:-4], 'file': os.path.join(
                                        recordings_location, recording), 'sentence': sentence.read(), 'severity': -1}
                                    df = df.append(new_row, ignore_index=True)
                                else:
                                    continue

            df['speech'] = [speech_file_to_array(x) for x in df['file']]

            dfs.append(Dataset.from_pandas(df))

        return dfs

    #------------------------------------------------------------------------


    # AUTOMATIC TESTING


    def automatic_test(dataset, local):
        if local:
            tds = import_torgo('/home/tim/Documents/Datasets/torgo/test')
            #test_dataset = import_torgo('/home/tim/Documents/Datasets/torgo/TORGO')
        else:
            tds = import_torgo('/work/herzig/datasets/torgo/TORGO')

        for test_dataset in tds:
            result = test_dataset.map(evaluate, batched=True, batch_size=8)

            predictions = [x.upper() for x in result["pred_text"]]
            references = [x.upper() for x in result["sentence"]]

            try:
                m_wer, m_cer = wer.compute(predictions=predictions, references=references, chunk_size=1000) * 100, cer.compute(predictions=predictions, references=references, chunk_size=1000) * 100
            except OSError: 
                pass

            print('Model: ' + str(MODEL_ID) + ', patient: ' + test_dataset[0]['id'])
            print('Dataset: ' + str(dataset))
            print('WER: ' + str(m_wer))
            print('CER: ' + str(m_cer))


    automatic_test(dataset, local)



for model in models_en:
    run_test(dataset, model, 'en', l)

#---------------------------------------------------------------------
