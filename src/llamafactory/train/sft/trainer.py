# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Configure FP8 environment if enabled
        if model_args is not None and model_args.fp8:
            configure_fp8_environment(model_args)
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self._use_segmented_loss = finetuning_args.thought_action_loss
        self._segment_markers: Optional[dict[str, list[int]]] = None
        self._thought_loss_weight = finetuning_args.thought_loss_weight
        self._action_loss_weight = finetuning_args.action_loss_weight

        if finetuning_args.use_dft_loss:
            if self._use_segmented_loss:
                raise ValueError("`thought_action_loss` cannot be combined with `use_dft_loss`.")

            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func
        elif self._use_segmented_loss:
            if self._thought_loss_weight < 0 or self._action_loss_weight < 0:
                raise ValueError("Loss weights must be non-negative.")
            if self._thought_loss_weight == 0 and self._action_loss_weight == 0:
                raise ValueError("At least one loss weight must be positive.")
            tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is None:
                tokenizer = getattr(self, "processing_class", None)
            if tokenizer is None and hasattr(self.data_collator, "tokenizer"):
                tokenizer = self.data_collator.tokenizer

            if tokenizer is None:
                raise RuntimeError("Tokenizer is required when `thought_action_loss=True`.")

            thought_tokens = tokenizer.encode(finetuning_args.thought_tag, add_special_tokens=False)
            action_tokens = tokenizer.encode(finetuning_args.action_tag, add_special_tokens=False)
            if len(thought_tokens) == 0 or len(action_tokens) == 0:
                raise ValueError("Unable to tokenize the provided thought/action markers.")

            self._segment_markers = {"thought": thought_tokens, "action": action_tokens}

        # Verify FP8 status after trainer initialization (accelerator should be available)
        if model_args is not None and model_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, model_args)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        if not getattr(self, "_use_segmented_loss", False):
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        labels: Optional[torch.Tensor] = inputs.get("labels")
        # Allow extra kwargs such as `num_items_in_batch` passed by Trainer.
        kwargs.pop("num_items_in_batch", None)
        outputs = model(**inputs)

        logits = None
        if isinstance(outputs, dict):
            logits = outputs.get("logits")
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 1:
            logits = outputs[1]

        if logits is None or labels is None:
            raise RuntimeError("`logits` and `labels` are required when `thought_action_loss=True`.")

        segment_map = self._build_segment_map(labels)
        loss = self._segment_average_loss(logits, labels, segment_map)

        if isinstance(outputs, dict):
            outputs["loss"] = loss
        elif hasattr(outputs, "loss"):
            outputs.loss = loss

        return (loss, outputs) if return_outputs else loss

    def _build_segment_map(self, labels: torch.Tensor) -> torch.Tensor:
        if self._segment_markers is None:
            return torch.zeros_like(labels, dtype=torch.long)

        segments = torch.zeros_like(labels, dtype=torch.long)
        thought_tokens = self._segment_markers["thought"]
        action_tokens = self._segment_markers["action"]

        label_rows = labels.detach().cpu().tolist()
        for row_idx, row in enumerate(label_rows):
            thought_start = self._find_subsequence(row, thought_tokens, 0)
            if thought_start == -1:
                continue

            after_thought = self._advance_past_pattern(row, thought_start, thought_tokens)
            action_start = self._find_subsequence(row, action_tokens, after_thought)

            last_valid = -1
            for pos in range(len(row) - 1, -1, -1):
                if row[pos] != IGNORE_INDEX:
                    last_valid = pos
                    break

            if last_valid == -1:
                continue

            current_segment = 1
            pos = thought_start
            while pos <= last_valid:
                token_id = row[pos]
                if token_id != IGNORE_INDEX:
                    if action_start != -1 and pos >= action_start:
                        current_segment = 2
                    segments[row_idx, pos] = current_segment
                pos += 1

        return segments.to(labels.device)

    @staticmethod
    def _find_subsequence(row: list[int], pattern: list[int], start: int) -> int:
        if not pattern:
            return -1

        plen = len(pattern)
        idx = start
        limit = len(row)
        while idx < limit:
            if row[idx] == IGNORE_INDEX:
                idx += 1
                continue

            j = idx
            k = 0
            while j < limit and k < plen:
                token = row[j]
                if token == IGNORE_INDEX:
                    j += 1
                    continue
                if token != pattern[k]:
                    break
                j += 1
                k += 1

            if k == plen:
                return idx

            idx += 1

        return -1

    @staticmethod
    def _advance_past_pattern(row: list[int], start: int, pattern: list[int]) -> int:
        idx = start
        matched = 0
        limit = len(row)
        while idx < limit and matched < len(pattern):
            token = row[idx]
            if token == IGNORE_INDEX:
                idx += 1
                continue
            if token != pattern[matched]:
                break
            idx += 1
            matched += 1

        return idx

    def _segment_average_loss(self, logits: torch.Tensor, labels: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_segments = segments[:, 1:].contiguous()

        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view_as(shift_labels)

        valid_mask = shift_labels.ne(IGNORE_INDEX)
        thought_mask = (shift_segments == 1) & valid_mask
        action_mask = (shift_segments == 2) & valid_mask

        # Compute weighted loss based on present segments
        weighted_loss = 0.0
        total_weight = 0.0

        if thought_mask.any():
            thought_loss = per_token_loss[thought_mask].mean()
            weighted_loss += thought_loss * self._thought_loss_weight
            total_weight += self._thought_loss_weight

        if action_mask.any():
            action_loss = per_token_loss[action_mask].mean()
            weighted_loss += action_loss * self._action_loss_weight
            total_weight += self._action_loss_weight

        # Fallback if no segments found
        if total_weight == 0:
            if valid_mask.any():
                return per_token_loss[valid_mask].mean()
            else:
                return per_token_loss.mean()

        # Normalize by total weight of present segments
        return weighted_loss / total_weight

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
