import onnxruntime as ort
import numpy as np
from transformers import BertTokenizer
from typing import List, Union


class Embedder:
    def __init__(
        self, model_dir: str = "./models/mobilebert-onnx", use_cpu: bool = False
    ):
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.session = ort.InferenceSession(
            f"{model_dir}/model.onnx",
            providers=["CPUExecutionProvider"] if use_cpu else ["QNNExecutionProvider"],
        )

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        tokens = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="np", max_length=128
        )

        ort_inputs = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }

        outputs = self.session.run(["pooler_output"], ort_inputs)
        return outputs[0]  # shape: (batch_size, hidden_size)
