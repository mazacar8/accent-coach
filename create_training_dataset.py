import pickle
from datasets import load_dataset, Audio, Features, Value
# logging.set_verbosity_debug()

SELECTED_SPEAKERS = ["v2/en_speaker_1", "v2/en_speaker_7", "v2/en_speaker_9", "v2/fr_speaker_2", "v2/fr_speaker_7", "original"]
SENTENCE_TO_SENTENCE_ID = pickle.load(open("new_sentence_to_sentence_id.pkl", "rb"))
INPUT_SPEAKERS = ["v2/en_speaker_1", "v2/en_speaker_7", "v2/en_speaker_9"]
OUTPUT_SPEAKERS = ["v2/fr_speaker_2", "v2/fr_speaker_7", "original"]

def create_training_dataset(batch):
    assert len(batch["sentence"]) == len(SELECTED_SPEAKERS), "Batch length must be length of SELECTED_SPEAKERS"
    sentence = batch["sentence"][0]
    speaker_to_audio = {speaker: audio for speaker, audio in zip(batch["speaker"], batch["audio"])}
    batch_results = {}
    for input_speaker in INPUT_SPEAKERS:
        for output_speaker in OUTPUT_SPEAKERS:
            batch_results.setdefault("sentence_id", []).append(SENTENCE_TO_SENTENCE_ID[sentence])
            batch_results.setdefault("sentence", []).append(sentence)
            batch_results.setdefault("input_speaker", []).append(input_speaker)
            batch_results.setdefault("output_speaker", []).append(output_speaker)
            batch_results.setdefault("source_audio", []).append(speaker_to_audio[input_speaker])
            batch_results.setdefault("target_audio", []).append(speaker_to_audio[output_speaker])

    return batch_results



def main():
    accent_audio_dataset = load_dataset("preetam8/accent_coach_all_waveforms", split="train")
    bark_inference_dataset = load_dataset("preetam8/bark_inference_for_accents", split="train")
    training_sentences = set(bark_inference_dataset.filter(lambda x: x["all_speakers_generated"])["sentence"])
    assert len(training_sentences) == 10000, f"Expected 10000 sentences but got {len(training_sentences)}"
    print("Starting filter")
    training_audios = accent_audio_dataset.filter(lambda x: x["sentence"] in training_sentences)
    assert len(training_audios) == 60000, f"Expected 60000 audios but got {len(training_audios)}"
    print("Starting map")
    accent_coach_training_dataset = training_audios.map(
        create_training_dataset, batched=True, batch_size=len(SELECTED_SPEAKERS), remove_columns=accent_audio_dataset.column_names
    )
    print(f"Length of accent_coach_training_dataset: {len(accent_coach_training_dataset)}")
    new_columns = Features({
        "sentence_id": Value("int32"),
        "sentence": Value("string"),
        "input_speaker": Value("string"),
        "output_speaker": Value("string"),
        "source_audio": Audio(sampling_rate=24_000),
        "target_audio": Audio(sampling_rate=24_000),
    })
    accent_coach_training_dataset = accent_coach_training_dataset.cast(new_columns)
    accent_coach_training_dataset = accent_coach_training_dataset.sort("sentence_id")
    accent_coach_training_dataset.push_to_hub("preetam8/accent_coach_training_dataset")


if __name__ == "__main__":
    main()