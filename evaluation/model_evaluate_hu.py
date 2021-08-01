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

dataset = 'hu'
models_de = ['jonatasgrosman/wav2vec2-large-xlsr-53-german', 'facebook/wav2vec2-large-xlsr-53-german']
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


    # Method to import HU datasets
    def import_hu(location):
        print('Import HU dataset')
        speakers = [x[0] for x in os.walk(location)]
        df = pd.DataFrame(columns=['id', 'file', 'sentence', 'severity'])

        speakers.pop(0)
        for speaker in speakers:
            files = os.listdir(speaker)
            file_path = ''
            transcription = ''
            severity = 0

            for file in files:
                if file[-12:] == 'severity.txt':
                    severity = int(open(os.path.join(speaker, file), 'r').read()) if len(
                        open(os.path.join(speaker, file), 'r').read()) > 0 else -1
                if file[-17:] == 'transcription.txt':
                    transcription = open(os.path.join(speaker, file), 'r').read()
                if file[-4:] == '.mp3':
                    file_path = os.path.join(speaker, file)

            new_row = {'id': speaker[6:], 'file': file_path,
                    'sentence': transcription, 'severity': severity}
            df = df.append(new_row, ignore_index=True)

        df['speech'] = [speech_file_to_array(x) for x in df['file']]
        return Dataset.from_pandas(df)

    #------------------------------------------------------------------------


    # AUTOMATIC TESTING


    def automatic_test(dataset, local):
        if local:
            test_dataset = import_hu('/home/tim/Documents/Datasets/hu_dysarthria/test')
            #test_dataset = import_hu('/home/tim/Documents/Datasets/hu_dysarthria/dysarthria')
        else:
            test_dataset = import_hu('/work/herzig/datasets/hu_dysarthria/dysarthria')

        result = test_dataset.map(evaluate, batched=True, batch_size=8)

        predictions = [x.upper() for x in result["pred_text"]]
        references = [x.upper() for x in result["sentence"]]

        m_wer, m_cer = wer.compute(predictions=predictions, references=references, chunk_size=1000) * 100, cer.compute(predictions=predictions, references=references, chunk_size=1000) * 100

        return m_wer, m_cer

    return automatic_test(dataset, local)


for model in models_de:
    wer, cer = run_test(dataset, model, 'de', l)
    print('Model: ' + str(model))
    print('Dataset: ' + str(dataset))
    print('WER: ' + str(wer))
    print('CER: ' + str(cer))

#---------------------------------------------------------------------
