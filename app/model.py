import json
import re
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


class SmolVLM:
    def __init__(self, model_name: str, max_new_tokens: int = 160):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.bfloat16
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            dtype = torch.float32
            print("WARNING: CUDA is not available; CPU inference will be very slow.")

        print(f"Loading {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device)

        self.model.eval()
        print("Model loaded.")

    def analyze(self, video_path: str, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            k: (v.to(self.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated = output_ids[:, input_len:]

        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0].strip()

    @staticmethod
    def parse_json(text: str) -> dict | None:
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        m = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            re.S | re.I,
        )
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        m = re.search(r"(\{.*\})", text, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        return None
