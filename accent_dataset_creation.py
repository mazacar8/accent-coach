import logging
import random
import torch

from datasets import concatenate_datasets
from datasets import Dataset
from datasets import load_dataset
from huggingface_hub import login
from transformers import BarkModel, BarkProcessor
from transformers.feature_extraction_utils import BatchFeature
from typing import Dict


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUGGINGFACE_TOKEN = "" # To be set/loaded securely
MODEL_NAME = "suno/bark-small"
SELECTED_SPEAKERS = ["v2/en_speaker_1", "v2/en_speaker_7", "v2/en_speaker_9", "v2/fr_speaker_2", "v2/fr_speaker_7"]
TOTAL_SENTENCES_PER_SPEAKER = 10000


def select_candidate_sentences(bark_inference_dataset, total_sentences_per_speaker):
    speaker_sentences = bark_inference_dataset.filter(lambda x: x["speaker"] == SELECTED_SPEAKERS[0])
    number_of_sentences_generated_all_speakers = len(speaker_sentences.filter(lambda x: x["all_speakers_generated"]))
    logging.info(f"Number of sentences already generated for all speakers: {number_of_sentences_generated_all_speakers}")
    remaining_to_select = total_sentences_per_speaker - number_of_sentences_generated_all_speakers
    generated_samples = bark_inference_dataset.filter(lambda x: x["train_data_generated"])
    sentence_to_num_speakers = {}
    for generated_sample in generated_samples:
       sentence = generated_sample["sentence"]
       sentence_to_num_speakers[sentence] = sentence_to_num_speakers.get(sentence, 0) + 1
    
    num_speakers_to_sentences = {}
    for sentence, num_speakers in sentence_to_num_speakers.items():
        num_speakers_to_sentences.setdefault(num_speakers, []).append(sentence)

    selected_sentences = set()
    for num_speakers in sorted(num_speakers_to_sentences, reverse=True):
       if remaining_to_select == 0:
          break
       num_to_select = min(remaining_to_select, len(num_speakers_to_sentences[num_speakers]))
       selected_sentences.update(num_speakers_to_sentences[num_speakers][:num_to_select])
       remaining_to_select -= num_to_select
       logging.info(f"Selected {num_to_select} sentences which have generated samples from {num_speakers} speakers")

    if remaining_to_select > 0:
       non_generated_samples = bark_inference_dataset.filter(lambda x: not x["train_data_generated"])
       selected_sentences.update(non_generated_samples.select(random.sample(range(len(non_generated_samples)), k=remaining_to_select)))

    return selected_sentences
    
    


def filter_dataset(bark_inference_dataset) -> Dict[str, Dataset]:
    datasets_split_by_speaker = {}
    selected_sentences = select_candidate_sentences(bark_inference_dataset, TOTAL_SENTENCES_PER_SPEAKER)
    for speaker in set(bark_inference_dataset["speaker"]):
        datasets_split_by_speaker[speaker] = bark_inference_dataset.filter(
           lambda x: x["speaker"] == speaker and x["sentence"] in selected_sentences and not x["train_data_generated"]
        )
        speaker_dataset_length = len(datasets_split_by_speaker[speaker])
        logging.info(f"Speaker: {speaker}, Dataset Length: {speaker_dataset_length}")
    return datasets_split_by_speaker


def initialize_model():
    logging.info(f"Using device: {DEVICE}")
    model = BarkModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
    model =  model.to_bettertransformer()
    processor = BarkProcessor.from_pretrained(MODEL_NAME)
    model.enable_cpu_offload()
    return model, processor


def move_dict_to_device(data_dict, device):
  for key in data_dict.keys():
    if isinstance(data_dict[key], torch.Tensor):
      data_dict[key] = data_dict[key].to(device)
    elif isinstance(data_dict[key], dict) or isinstance(data_dict[key], BatchFeature):
      data_dict[key] = move_dict_to_device(data_dict[key], device)
  return data_dict


def update_train_data_generated(example, target_speaker, updated_sentences):
   if example["train_data_generated"] or (example["speaker"] == target_speaker and example["sentence"] in updated_sentences):
      return {"train_data_generated": True}
   return {"train_data_generated": False}


def determine_if_all_speakers_generated(batch):
    if len(set(batch["sentence"])) > 1:
        raise Exception(f"Batch size has to be length of SELECTED_SPEAKERS and dataset has to be sorted by sentence")

    if not all(batch["train_data_generated"]):
        return {"all_speakers_generated": [False]*len(batch["sentence"])}
   
    return {"all_speakers_generated": [True]*len(batch["sentence"])}


def main():
    """
    Main function to process and generate datasets for accent correction.

    This function performs the following steps:
    1. Authenticates with the Hugging Face Hub using a token.
    2. Loads the bark inference dataset and filters it by speaker.
    3. Initializes the Bark model and processor.
    4. Defines a helper function to generate audio using the Bark model.
    5. Iterates over datasets split by speaker, generating audio and pushing datasets to the Hugging Face Hub.
    6. Updates the bark inference dataset with generation status and pushes changes to the Hub.
    7. Concatenates generated datasets and existing datasets, sorts, and pushes the final dataset to the Hub.
    """
    login(token=HUGGINGFACE_TOKEN)
    bark_inference_dataset = load_dataset("preetam8/bark_inference_for_accents", split="train")
    datasets_split_by_speaker = filter_dataset(bark_inference_dataset)
    model, processor = initialize_model()

    def generate_bark_audio(batch):
        batch_inputs = processor(batch["sentence"], voice_preset=batch["speaker"][0])
        batch_inputs = move_dict_to_device(batch_inputs, DEVICE)
        batch_speech_outputs = model.generate(**batch_inputs).cpu().numpy()
        return {
            "waveform": batch_speech_outputs,
        }
    
    datasets_with_audio = {}
    for speaker, dataset in datasets_split_by_speaker.items():
        logging.info(f"Starting dataset creation for {speaker}")
        try:
            speaker_dataset = dataset.map(
                generate_bark_audio,
                batched=True,
                batch_size=32,
                remove_columns=["train_data_generated", "all_speakers_generated"]
            )
        except Exception as e:
            logging.error(f"Failed with exception {e}")
        datasets_with_audio[speaker] = speaker_dataset
        short_speaker_name = speaker[speaker.find("/")+1:]
        speaker_dataset.push_to_hub(f"preetam8/accent_correction_dataset_v2_{short_speaker_name}")
        updated_sentences = set(dataset["sentence"])
        bark_inference_dataset = bark_inference_dataset.map(lambda x: update_train_data_generated(x, speaker, updated_sentences))
        bark_inference_dataset.push_to_hub("preetam8/bark_inference_for_accents")

    accent_dataset = concatenate_datasets(list(datasets_with_audio.values()))
    accent_dataset.push_to_hub("preetam8/accent_correction_dataset")

    bark_inference_dataset = bark_inference_dataset.map(determine_if_all_speakers_generated, batched=True, batch_size=len(SELECTED_SPEAKERS))
    bark_inference_dataset.push_to_hub("preetam8/bark_inference_for_accents")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
