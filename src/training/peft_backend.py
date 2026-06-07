from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.io_utils import read_jsonl


REQUIRED_MODULES = ("torch", "transformers", "datasets", "peft", "accelerate")


def missing_dependencies() -> list[str]:
    missing = []
    for module_name in REQUIRED_MODULES:
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    return missing


@dataclass
class PeftTrainingConfig:
    dataset_path: Path
    output_dir: Path
    run_name: str
    student_model: str
    teacher_model: str | None = None
    teacher_device: str | None = None
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_length: int = 512
    distill_alpha: float = 0.0
    distill_temperature: float = 1.0
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def _load_examples(dataset_path: Path) -> list[dict]:
    return list(read_jsonl(dataset_path))


def _resolve_dtype(torch_module) -> object:
    return torch_module.bfloat16 if torch_module.cuda.is_available() else torch_module.float32


def run_peft_training(config: PeftTrainingConfig) -> dict:
    missing = missing_dependencies()
    if missing:
        raise RuntimeError(
            "Missing training dependencies: "
            + ", ".join(missing)
            + ". Install the PEFT stack before running real training."
        )

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    import torch.nn.functional as F

    examples = _load_examples(config.dataset_path)
    if not examples:
        raise ValueError(f"No training examples found in {config.dataset_path}")

    use_distillation = config.teacher_model is not None and config.distill_alpha > 0.0

    tokenizer = AutoTokenizer.from_pretrained(config.student_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list(
        [
            {"text": f"{example['prompt']}\n{example['response']}"}
            for example in examples
        ]
    )

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.max_length,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    model = AutoModelForCausalLM.from_pretrained(
        config.student_model,
        dtype=_resolve_dtype(torch),
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    teacher_model = None
    teacher_device = None
    if use_distillation:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            config.teacher_model,
            dtype=_resolve_dtype(torch),
        )
        teacher_model.eval()
        if torch.cuda.is_available():
            teacher_device = torch.device(config.teacher_device or "cuda")
            teacher_model.to(teacher_device)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        bf16=torch.cuda.is_available(),
        fp16=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    class DistillationTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs["labels"]
            student_outputs = model(**inputs)
            hard_loss = student_outputs.loss
            loss = hard_loss
            if teacher_model is not None:
                teacher_inputs = {
                    "input_ids": inputs["input_ids"].to(teacher_device),
                    "attention_mask": inputs["attention_mask"].to(teacher_device),
                }
                with torch.no_grad():
                    teacher_outputs = teacher_model(**teacher_inputs)

                student_logits = student_outputs.logits
                teacher_logits = teacher_outputs.logits.to(student_logits.device)
                temperature = config.distill_temperature
                mask = labels.ne(-100)
                if mask.any():
                    student_log_probs = F.log_softmax(student_logits[mask] / temperature, dim=-1)
                    teacher_probs = F.softmax(teacher_logits[mask] / temperature, dim=-1)
                    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
                    loss = ((1.0 - config.distill_alpha) * hard_loss) + (config.distill_alpha * soft_loss)
            return (loss, student_outputs) if return_outputs else loss

    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    if teacher_device is not None:
        trainer.args._n_gpu = 1
    training_result = trainer.train()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    torch.save(
        {
            "run_name": config.run_name,
            "student_model": config.student_model,
            "adapter_state_dict": get_peft_model_state_dict(model),
        },
        config.output_dir / "model_final.pth",
    )

    metrics = {
        key: value
        for key, value in training_result.metrics.items()
        if isinstance(value, (int, float))
    }

    return {
        "run_name": config.run_name,
        "output_dir": str(config.output_dir),
        "num_examples": len(examples),
        "training_backend": "peft_lora_distill" if use_distillation else "peft_lora",
        "status": "completed",
        "metrics": metrics,
    }
