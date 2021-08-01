import re
import os
from datasets.arrow_dataset import DatasetTransformationNotAllowedError
import torch
import librosa
import warnings
import pandas as pd
import soundfile as sf

from datasets import load_dataset, load_metric, Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoTokenizer, AutoModel

# Specify which
#     - Datasets
#     - Models both English and German
#     - Is the test run local or in the cluster

dataset = 'mls_en'
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


    def map_to_array(batch):
        speech, _ = sf.read(batch['file'], samplerate=16000)
        batch['speech'] = speech
        return batch


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


    #------------------------------------------------------------------------


    # AUTOMATIC TESTING


    def automatic_test(dataset, local):
        test_dataset = load_dataset('librispeech_asr', 'clean', split='test')
        test_dataset = test_dataset.map(map_to_array)

        result = test_dataset.map(evaluate, batched=True, batch_size=8)

        predictions = [x.upper() for x in result["pred_text"]]
        references = [x.upper() for x in result["sentence"]]
        
        m_wer, m_cer = wer.compute(predictions=predictions, references=references, chunk_size=1000) * 100, cer.compute(predictions=predictions, references=references, chunk_size=1000) * 100

        return m_wer, m_cer

    return automatic_test(dataset, local)


for model in models_en:
    wer, cer = run_test(dataset, model, 'en', l)
    print('Model: ' + str(model))
    print('Dataset: ' + str(dataset))
    print('WER: ' + str(wer))
    print('CER: ' + str(cer))

#---------------------------------------------------------------------
