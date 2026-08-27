import json
import re
import torch
import av
import numpy as np
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

    def _load_video_frames(self, video_path: str, max_frames: int = 32):
        """
        Load video frames using av (PyAV) instead of torchcodec.
        This avoids the torchcodec DLL error completely.
        """
        container = av.open(video_path)
        stream = container.streams.video[0]

        # Calculate frame sampling rate to get ~max_frames evenly distributed
        total_frames = stream.frames
        if total_frames > max_frames:
            step = total_frames // max_frames
        else:
            step = 1

        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i % step == 0 and len(frames) < max_frames:
                img = frame.to_ndarray(format="rgb24")
                frames.append(img)

        container.close()
        return frames

    def analyze(self, video_path: str, prompt: str) -> str:
        # Load video frames manually using av (PyAV)
        video_frames = self._load_video_frames(video_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "frames": video_frames},
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

    def generate_text(
        self,
        system: str,
        user: str,
        max_new_tokens: int | None = None,
    ) -> str:
        """Text-only chat completion. Used by Stage 2 (derive highlight
        types) where there's no video to attach.
        """
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
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
                do_sample=True,
                temperature=0.7,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
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