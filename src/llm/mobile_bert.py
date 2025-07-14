from transformers import BertTokenizer, MobileBertModel
import torch
from pathlib import Path

model_name = "google/mobilebert-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = MobileBertModel.from_pretrained(model_name)
model.eval()

sample_input = tokenizer(
    "This is a test input.",
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=128,
)

onnx_path = Path("models/mobilebert-onnx")
onnx_path.mkdir(parents=True, exist_ok=True)

torch.onnx.export(
    model,
    (sample_input["input_ids"], sample_input["attention_mask"]),
    str(onnx_path / "model.onnx"),
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state", "pooler_output"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "last_hidden_state": {0: "batch_size", 1: "seq_len"},
        "pooler_output": {0: "batch_size"},
    },
    opset_version=13,
)

tokenizer.save_pretrained(onnx_path)
print(f"Exported MobileBERT to ONNX at: {onnx_path}")
