from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def require_training_dependencies() -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as functional
        from torch.utils.data import Dataset
        from transformers import Trainer
    except ImportError as exc:
        raise RuntimeError(
            "Student training dependencies are missing. Install "
            "requirements-student.txt on the GPU training server."
        ) from exc
    return {
        "torch": torch,
        "functional": functional,
        "Dataset": Dataset,
        "Trainer": Trainer,
    }


@dataclass
class LossConfig:
    method: str
    contrastive_weight: float = 0.1
    structure_weight: float = 0.2
    contrastive_temperature: float = 0.07
    structure_margin: float = 0.5


def build_runtime_classes():
    deps = require_training_dependencies()
    torch = deps["torch"]
    functional = deps["functional"]
    Dataset = deps["Dataset"]
    Trainer = deps["Trainer"]

    class StudentDataset(Dataset):
        def __init__(
            self,
            records,
            tokenizer,
            max_length: int,
            max_target_length: int,
        ):
            self.records = records
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.max_target_length = max_target_length
            keys = sorted({record["contrastive_key"] for record in records})
            self.class_ids = {key: index for index, key in enumerate(keys)}

        def __len__(self):
            return len(self.records)

        def _encode(self, record, target):
            messages = [
                {"role": "system", "content": record["system"]},
                {"role": "user", "content": record["prompt"]},
            ]
            try:
                prompt_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            target_ids = self.tokenizer(
                target,
                add_special_tokens=False,
            )["input_ids"] + [self.tokenizer.eos_token_id]
            if len(target_ids) > self.max_target_length:
                raise ValueError(
                    f"Encoded target has {len(target_ids)} tokens, exceeding "
                    f"max_target_length={self.max_target_length}. Increase the "
                    "limit instead of training on truncated invalid JSON."
                )
            prompt_budget = self.max_length - len(target_ids)
            if prompt_budget <= 0:
                raise ValueError(
                    "max_length must be larger than the encoded target length"
                )
            if len(prompt_ids) > prompt_budget:
                prefix = min(128, prompt_budget // 3)
                prompt_ids = (
                    prompt_ids[:prefix]
                    + prompt_ids[-(prompt_budget - prefix) :]
                )
            input_ids = prompt_ids + target_ids
            prompt_length = len(prompt_ids)
            labels = [-100] * prompt_length + input_ids[prompt_length:]
            return input_ids, labels, prompt_length

        def __getitem__(self, index):
            record = self.records[index]
            input_ids, labels, prompt_length = self._encode(
                record, record["target"]
            )
            item = {
                "input_ids": input_ids,
                "labels": labels,
                "prompt_length": prompt_length,
                "sample_weight": record["sample_weight"],
                "contrastive_label": self.class_ids[record["contrastive_key"]],
            }
            if record.get("negative_target"):
                negative_ids, negative_labels, _ = self._encode(
                    record, record["negative_target"]
                )
                item["negative_input_ids"] = negative_ids
                item["negative_labels"] = negative_labels
            return item

    class StudentCollator:
        def __init__(self, tokenizer):
            self.pad_token_id = tokenizer.pad_token_id

        @staticmethod
        def _pad(rows, value):
            width = max(len(row) for row in rows)
            return [row + [value] * (width - len(row)) for row in rows]

        def __call__(self, features):
            batch = {
                "input_ids": torch.tensor(
                    self._pad([x["input_ids"] for x in features], self.pad_token_id)
                ),
                "labels": torch.tensor(
                    self._pad([x["labels"] for x in features], -100)
                ),
                "sample_weight": torch.tensor(
                    [x["sample_weight"] for x in features], dtype=torch.float
                ),
                "contrastive_label": torch.tensor(
                    [x["contrastive_label"] for x in features], dtype=torch.long
                ),
                "prompt_length": torch.tensor(
                    [x["prompt_length"] for x in features], dtype=torch.long
                ),
            }
            batch["attention_mask"] = batch["input_ids"].ne(self.pad_token_id).long()
            if any("negative_input_ids" in feature for feature in features):
                negative_features = []
                has_negative = []
                for feature in features:
                    if "negative_input_ids" in feature:
                        negative_features.append(
                            (
                                feature["negative_input_ids"],
                                feature["negative_labels"],
                            )
                        )
                        has_negative.append(True)
                    else:
                        negative_features.append(
                            (feature["input_ids"], feature["labels"])
                        )
                        has_negative.append(False)
                batch["negative_input_ids"] = torch.tensor(
                    self._pad(
                        [item[0] for item in negative_features],
                        self.pad_token_id,
                    )
                )
                batch["negative_labels"] = torch.tensor(
                    self._pad([item[1] for item in negative_features], -100)
                )
                batch["has_negative"] = torch.tensor(
                    has_negative, dtype=torch.bool
                )
            return batch

    def per_sample_nll(logits, labels):
        shifted_logits = logits[:, :-1].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        token_losses = functional.cross_entropy(
            shifted_logits.transpose(1, 2),
            shifted_labels,
            ignore_index=-100,
            reduction="none",
        )
        mask = shifted_labels.ne(-100)
        token_counts = mask.sum(dim=1).clamp_min(1)
        return (token_losses * mask).sum(dim=1) / token_counts

    def supervised_two_view_contrastive_loss(
        left, right, labels, temperature
    ):
        left = functional.normalize(left, dim=-1)
        right = functional.normalize(right, dim=-1)
        logits = left @ right.transpose(0, 1) / temperature
        positive_mask = labels[:, None].eq(labels[None, :])

        def directional_loss(scores, mask):
            log_prob = scores - torch.logsumexp(scores, dim=1, keepdim=True)
            positive_count = mask.sum(dim=1).clamp_min(1)
            return -(
                (log_prob * mask).sum(dim=1) / positive_count
            ).mean()

        return 0.5 * (
            directional_loss(logits, positive_mask)
            + directional_loss(logits.transpose(0, 1), positive_mask.transpose(0, 1))
        )

    class StructuredKDTrainer(Trainer):
        def __init__(self, *args, loss_config: LossConfig, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_config = loss_config
            self.model_accepts_loss_kwargs = False

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            sample_weight = inputs.pop("sample_weight")
            contrastive_label = inputs.pop("contrastive_label")
            prompt_length = inputs.pop("prompt_length")
            labels = inputs.pop("labels")
            negative_input_ids = inputs.pop("negative_input_ids", None)
            negative_labels = inputs.pop("negative_labels", None)
            has_negative = inputs.pop("has_negative", None)
            need_representation = model.training and self.loss_config.method in {
                "contrastive_kd",
                "ours",
            } and self.loss_config.contrastive_weight > 0
            captured_hidden = {}
            final_norm = getattr(getattr(model, "model", None), "norm", None)
            hook = None
            if need_representation and final_norm is not None:
                hook = final_norm.register_forward_hook(
                    lambda module, args, output: captured_hidden.update(
                        {"value": output}
                    )
                )
            elif need_representation:
                inputs["output_hidden_states"] = True
            try:
                outputs = model(**inputs)
            finally:
                if hook is not None:
                    hook.remove()
            positive_nll = per_sample_nll(outputs.logits, labels)
            sample_weight = torch.nan_to_num(
                sample_weight.float(), nan=1.0, posinf=1.0, neginf=1.0
            ).clamp_min(0.1)
            # Keep weights absolute. Normalizing by their sum would cancel a
            # low teacher weight whenever a mini-batch contains only teacher
            # samples, which is common with a small Gold set.
            loss = (positive_nll * sample_weight).mean()

            if (
                model.training
                and self.loss_config.method in {"contrastive_kd", "ours"}
                and self.loss_config.contrastive_weight > 0
            ):
                hidden = captured_hidden.get("value")
                if hidden is None:
                    hidden = outputs.hidden_states[-1]
                token_positions = torch.arange(
                    hidden.size(1), device=hidden.device
                )[None, :]
                prompt_mask = token_positions.lt(prompt_length[:, None])
                prompt_mask = prompt_mask & inputs["attention_mask"].bool()
                first_view = (
                    hidden * prompt_mask.unsqueeze(-1)
                ).sum(dim=1) / prompt_mask.sum(dim=1, keepdim=True).clamp_min(1)
                left_view = functional.dropout(
                    first_view, p=0.1, training=True
                )
                right_view = functional.dropout(
                    first_view, p=0.1, training=True
                )
                contrastive = supervised_two_view_contrastive_loss(
                    left_view,
                    right_view,
                    contrastive_label,
                    self.loss_config.contrastive_temperature,
                )
                loss = loss + self.loss_config.contrastive_weight * contrastive

            if (
                model.training
                and self.loss_config.method == "ours"
                and self.loss_config.structure_weight > 0
                and negative_input_ids is not None
                and negative_labels is not None
            ):
                negative_attention = negative_input_ids.ne(
                    self.data_collator.pad_token_id
                ).long()
                negative_outputs = model(
                    input_ids=negative_input_ids,
                    attention_mask=negative_attention,
                )
                negative_nll = per_sample_nll(
                    negative_outputs.logits, negative_labels
                )
                structure_losses = functional.relu(
                    self.loss_config.structure_margin
                    + positive_nll
                    - negative_nll
                )
                if has_negative is not None and has_negative.any():
                    structure_loss = (
                        structure_losses[has_negative]
                        * sample_weight[has_negative]
                    ).mean()
                    loss = (
                        loss
                        + self.loss_config.structure_weight * structure_loss
                    )
            return (loss, outputs) if return_outputs else loss

    return StudentDataset, StudentCollator, StructuredKDTrainer
