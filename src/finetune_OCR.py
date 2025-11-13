from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit"   # model base 4bit
OUTPUT_DIR = "models/unsloth_training_logs"            # log/checkpoint tạm
FINAL_MODEL_DIR = "models/unsloth_ocr"                 # nơi lưu model đã merge
SEED = 42
MAX_STEPS = 50
DEVICE = "cuda"                                        # "cuda" hoặc "cpu" (cpu rất chậm)

INSTRUCTION = """
You are a world-class OCR expert specializing in recognizing all types of vehicle license plates
(cars, motorbikes, trucks, etc.) in any weather or lighting condition, including blurred, dirty,
or low-contrast images. Your recognition must be precise and avoid any confusion between
similar-looking characters (e.g., '0' and 'O', '1' and 'I', '8' and 'B').

Analyze the given image, which may contain one or multiple license plates.
For each license plate detected, extract and return ONLY its exact content,
using only the following valid characters: digits (0-9), uppercase letters (A-Z),
the hyphen (-), and the dot (.).

List each license plate you find on a separate line, with no extra words,
symbols, or explanations.
"""


def load_raw_dataset():
    dataset = load_dataset("EZCon/taiwan-license-plate-recognition", split="train")
    dataset = dataset.remove_columns(["xywhr", "is_electric_car"])
    dataset = dataset.rename_column("license_number", "text")
    return dataset


def convert_sample_to_conversation(sample):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": INSTRUCTION},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["text"]},
                ],
            },
        ]
    }


def get_converted_dataset():
    raw_dataset = load_raw_dataset()
    converted_dataset = [convert_sample_to_conversation(s) for s in raw_dataset]
    return converted_dataset



def load_model_and_tokenizer():
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=SEED,
    )

    return model, tokenizer


def main():
    # Dataset
    converted_dataset = get_converted_dataset()

    # Model + tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Chuyển sang chế độ training
    FastVisionModel.for_training(model)

    # Tạo trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted_dataset,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=MAX_STEPS,
            learning_rate=2e-4,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=SEED,
            output_dir=OUTPUT_DIR,
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=2048,
        ),
    )

    # Train
    trainer.train()

    # Lưu model đã merge để dùng inference sau này
    model.save_pretrained_merged(FINAL_MODEL_DIR, tokenizer)


if __name__ == "__main__":
    main()
