
# LoRA Fine-Tuning with Incremental Chunk Training (Qwen3 + Colab T4)

##  Overview
This project demonstrates efficient fine-tuning of a large language model using LoRA with chunk-based training on a large dataset using **Google Colab (T4 GPU)**.

## Dataset
- Original Train: 287,113 samples
- Subset Used: 80,000 samples (after shuffling)

## Data Strategy
```python
train_dataset = tokenized_dataset["train"].shuffle(seed=42)
train_dataset = train_dataset.select(range(80000))
```

## Incremental Training
- Chunk size: 8,000 samples
- Total chunks: 10

## Chunking
```python
chunk_size = 8000
for i in range(0, len(train_dataset), chunk_size):
    chunk = train_dataset.select(range(i, i + chunk_size))
    chunk.save_to_disk(f"/content/drive/MyDrive/cnn_chunks/chunk_{i//chunk_size}")
```

## Workflow
Chunk → Train → Save → Load → Next Chunk

## Model Setup
- 4-bit quantization (bitsandbytes)
- LoRA applied to base model

## Training Config
- Batch size: 2
- Gradient Accumulation: 8
- Epochs: 1

## Save LoRA
```python
trainer.model.save_pretrained("/content/drive/MyDrive/qwen3_lora_progress")
```

## Load LoRA
```python
model = PeftModel.from_pretrained(base_model, path, is_trainable=True)
```

## Key Fixes
- Use `is_trainable=True`
- Save to Drive (not `/content`)
- Reinitialize Trainer every chunk

## Outcome
- Efficient training on Colab T4
- Robust incremental pipeline

## License
Educational use only
