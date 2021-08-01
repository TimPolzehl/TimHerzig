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

dataset = 'mls_de'
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

    def split_handle(handle, audio_location):
        speaker_id, book_id, audio_id = handle.split('_')
        return os.path.join(audio_location, speaker_id, book_id, handle + '.opus')


    # Method to import Multilingual Librispeech datasets (handles.txt files)
    # Returns PD-Dataframe with 'id', 'sentence', 'location'
    def import_mls(handle_location, transcription_location, audio_location):
        print('Import MLS dataset')

        h = pd.read_csv(handle_location, delimiter="\t", header=None)
        if(len(h.columns) >= 2):
            h.drop(h.columns[[1, 2, 3]], axis=1, inplace=True)
        ts = pd.read_csv(transcription_location, delimiter="\t", header=None)

        ts.columns = ['id', 'sentence']
        h.columns = ['id']

        df = pd.merge(ts, h, left_on='id', right_on='id')
        df['file'] = [split_handle(x, audio_location) for x in df['id']]
        df['speech'] = [speech_file_to_array(x) for x in df['file']]

        return Dataset.from_pandas(df)

    #------------------------------------------------------------------------


    # AUTOMATIC TESTING


    def automatic_test(dataset, local):
        if local:
            # test_dataset = import_mls(
            #     '/home/tim/Documents/Datasets/mls_german_opus/test/segments.txt',
            #     '/home/tim/Documents/Datasets/mls_german_opus/test/transcripts.txt',
            #     '/home/tim/Documents/Datasets/mls_german_opus/test/audio/')

            test_dataset = import_mls(
                '/home/tim/Documents/Datasets/mls_german_opus/train/limited_supervision/1hr/0/handles.txt',
                '/home/tim/Documents/Datasets/mls_german_opus/train/transcripts.txt',
                '/home/tim/Documents/Datasets/mls_german_opus/train/audio/')
        else:
            test_dataset = import_mls(
                '/work/herzig/datasets/mls_german_opus/test/segments.txt',
                '/work/herzig/datasets/mls_german_opus/test/transcripts.txt',
                '/work/herzig/datasets/mls_german_opus/test/audio/')

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
