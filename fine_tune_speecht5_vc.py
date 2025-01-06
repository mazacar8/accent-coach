import logging
import numpy as np
import torch

from datasets import DatasetDict
from datasets import Features
from datasets import load_dataset
from datasets import Sequence
from datasets import Value
from huggingface_hub import login
from torch.nn.utils.rnn import pad_sequence
from transformers import SpeechT5ForSpeechToSpeech
from transformers import SpeechT5Processor
from transformers import Trainer
from transformers import TrainingArguments
from transformers.models.speecht5.modeling_speecht5 import shift_spectrograms_right
from transformers.models.speecht5.modeling_speecht5 import SpeechT5SpectrogramLoss

from typing import Optional, Tuple, Union

from transformers.modeling_outputs import Seq2SeqSpectrogramOutput


TEST_RUN = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
OUTPUT_DTYPE = torch.float32
ATTENTION_DTYPE = torch.long
HUGGINGFACE_TOKEN = "" # To be set/loaded securely

TRAINING_ARGUMENTS = {
    "default": {
        "output_dir": "./speecht5_vc_finetuned_accent_coach",
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "logging_dir": "./logs",
        "report_to": "tensorboard",
    },
    "cuda": {
        "per_device_train_batch_size": 64,
        "per_device_eval_batch_size": 64,
        "num_train_epochs": 10,
        "logging_steps": 375,
        "save_steps": 1125,
        "eval_steps": 1125,
        "fp16": True,
        "use_cpu": False,
        "push_to_hub": True,
        "hub_model_id": "preetam8/speecht5_vc_finetuned_accent_coach",
        "hub_strategy": "checkpoint",
        "hub_token": HUGGINGFACE_TOKEN,
        "dataloader_pin_memory": False,
        "label_names": ["labels"],
        "load_best_model_at_end": True,
        "metric_for_best_model": "loss",
        "greater_is_better": False,
        "eval_on_start": True,
    },
    "cpu": {
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "num_train_epochs": 1,
        "logging_steps": 2,
        "save_steps": 100,
        "eval_steps": 100,
        "fp16": False,
        "use_cpu": True,
    }
}


class SpeechT5ForSpeechToSpeechForTraining(SpeechT5ForSpeechToSpeech):

    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_values: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        stop_labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Seq2SeqSpectrogramOutput]:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
            into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile library (*pip install
            soundfile*). To prepare the array into `input_values`, the [`SpeechT5Processor`] should be used for padding
            and conversion into a tensor of type `torch.FloatTensor`. See [`SpeechT5Processor.__call__`] for details.
        decoder_input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_mel_bins)`):
            Float values of input mel spectrogram.

            SpeechT5 uses an all-zero spectrum as the starting token for `decoder_input_values` generation. If
            `past_key_values` is used, optionally only the last `decoder_input_values` have to be input (see
            `past_key_values`).
        speaker_embeddings (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_dim)`, *optional*):
            Tensor containing the speaker embeddings.
        labels (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_mel_bins)`, *optional*):
            Float values of target mel spectrogram. Spectrograms can be obtained using [`SpeechT5Processor`]. See
            [`SpeechT5Processor.__call__`] for details.

        Returns:

        Example:

        ```python
        >>> from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan, set_seed
        >>> from datasets import load_dataset
        >>> import torch

        >>> dataset = load_dataset(
        ...     "hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True
        ... )  # doctest: +IGNORE_RESULT
        >>> dataset = dataset.sort("id")
        >>> sampling_rate = dataset.features["audio"].sampling_rate

        >>> processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
        >>> model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
        >>> vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        >>> # audio file is decoded on the fly
        >>> inputs = processor(audio=dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

        >>> speaker_embeddings = torch.zeros((1, 512))  # or load xvectors from a file

        >>> set_seed(555)  # make deterministic

        >>> # generate speech
        >>> speech = model.generate_speech(inputs["input_values"], speaker_embeddings, vocoder=vocoder)
        >>> speech.shape
        torch.Size([77824])
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if decoder_input_values is None:
                decoder_input_values, decoder_attention_mask = shift_spectrograms_right(
                    labels, self.config.reduction_factor, decoder_attention_mask
                )
            if self.config.use_guided_attention_loss:
                output_attentions = True

        outputs = self.speecht5(
            input_values=input_values,
            attention_mask=attention_mask,
            decoder_input_values=decoder_input_values,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            speaker_embeddings=speaker_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )


        outputs_before_postnet, spectrogram, logits = self.speech_decoder_postnet(outputs[0])

        loss = None
        if labels is not None:
            criterion = SpeechT5SpectrogramLoss(self.config)
            fv_attention_mask = self.speecht5.encoder.prenet._get_feature_vector_attention_mask(
                outputs.encoder_last_hidden_state.shape[1], attention_mask
            )
            loss = criterion(
                fv_attention_mask,
                outputs_before_postnet,
                spectrogram,
                logits,
                labels,
                outputs.cross_attentions
            )

        if not return_dict:
            output = (spectrogram,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSpectrogramOutput(
            loss=loss,
            spectrogram=spectrogram,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
    
class DataCollatorWithPadding:

    def __call__(self, features):
        input_values = [f["input_values"] for f in features]
        labels = [f["labels"] for f in features]

        input_values = pad_sequence(input_values, batch_first=True, padding_value=-100).to(dtype=INPUT_DTYPE, device=DEVICE)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(dtype=OUTPUT_DTYPE, device=DEVICE)

        if labels.shape[1] % 2 == 1:
            pad = (torch.ones((labels.shape[0], 1, labels.shape[2]))*-100).to(dtype=OUTPUT_DTYPE, device=DEVICE)
            labels = torch.cat((labels, pad), dim=1)

        attention_mask = (input_values != -100).to(dtype=ATTENTION_DTYPE, device=DEVICE)
        decoder_attention_mask = (labels != -100).to(dtype=ATTENTION_DTYPE, device=DEVICE)

        return {
            "input_values": input_values,
            "labels":labels,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
        }


def preprocess_dataset(source_audio, target_audio, processor):
    processor_output = processor(
        audio=source_audio["array"],
        audio_target=target_audio["array"],
        sampling_rate=16_000,
        return_attention_mask=False,
    )
    input_values = np.array(processor_output.input_values[0], dtype=np.float16)
    labels = np.array(processor_output.labels[0], dtype=np.float32)
    return {"input_values": input_values, "labels": labels}


def get_final_dataset(accent_coach_dataset):
    final_dataset = {}
    features = Features({
        "input_values": Sequence(Value(dtype="float16")),
        "labels": Sequence(Sequence(Value(dtype="float32"))),
    })
    for split in accent_coach_dataset:
        final_dataset[split] = accent_coach_dataset[split].map(
            preprocess_dataset,
            input_columns=["source_audio", "target_audio"],
            features=features,
            remove_columns=accent_coach_dataset[split].column_names,
            fn_kwargs={"processor": SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")},
            keep_in_memory=True,
        )

    final_dataset = DatasetDict(final_dataset)
    return final_dataset


def initialize_model():
    model = SpeechT5ForSpeechToSpeechForTraining.from_pretrained("microsoft/speecht5_vc")
    model = model.to(device=DEVICE)
    model = model.train()
    return model


def main():
    login(token=HUGGINGFACE_TOKEN)
    logging.info(f"Using device: {DEVICE}, Input dtype: {INPUT_DTYPE}, Output dtype: {OUTPUT_DTYPE}")
    accent_coach_dataset = load_dataset("preetam8/accent_coach_training_dataset")
    final_dataset = get_final_dataset(accent_coach_dataset)


    train_dataset = final_dataset["train"].with_format("torch")
    validation_dataset = final_dataset["validation"].with_format("torch")

    model = initialize_model()
    logging.info(model.config)
    
    data_collator = DataCollatorWithPadding()

    device_args = TRAINING_ARGUMENTS["cuda"] if (DEVICE == "cuda" and not TEST_RUN) else TRAINING_ARGUMENTS["cpu"]
    training_args = TrainingArguments(**TRAINING_ARGUMENTS["default"], **device_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()