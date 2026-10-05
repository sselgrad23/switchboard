"""Chat-model backends behind one small interface.

``QwenChat`` runs Qwen2.5-3B-Instruct locally with ``transformers`` and passes the
tool schemas through the model's own chat template (``apply_chat_template(tools=)``),
which renders them in the format the model was trained on for function calling.
``ScriptedChat`` replays fixed replies so the agent loop can be unit-tested without
a GPU.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from switchboard import config

logger = logging.getLogger(__name__)


@dataclass
class Completion:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0


class ChatModel(Protocol):
    name: str

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Completion: ...


class QwenChat:
    """Local Qwen2.5-3B-Instruct. Greedy decoding, so evaluation runs are repeatable."""

    def __init__(self, model_name: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name or config.LLM_MODEL
        kwargs: dict[str, Any] = {"device_map": "auto"}
        if config.LLM_LOAD_4BIT:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            # fp16, not bf16: Turing GPUs (RTX 20xx) have no bf16 support.
            kwargs["torch_dtype"] = torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, **kwargs)
        self.model.eval()

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Completion:
        import torch

        text = self.tokenizer.apply_chat_template(
            messages, tools=tools or None, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        n_in = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=config.LLM_MAX_NEW_TOKENS, do_sample=False,
                temperature=None, top_p=None, top_k=None,
            )
        gen = out[0][n_in:]
        del inputs
        # Release cached blocks between calls: on an 8 GB card under WSL, a growing
        # cache spills into shared system memory and generation slows several-fold.
        torch.cuda.empty_cache()
        return Completion(
            text=self.tokenizer.decode(gen, skip_special_tokens=True),
            tokens_in=n_in, tokens_out=int(gen.shape[0]),
        )


class ScriptedChat:
    """Test double: returns canned replies in order (or computes them from messages)."""

    name = "scripted"

    def __init__(self, replies: list[str] | Callable[[list[dict[str, Any]]], str]) -> None:
        self._replies = replies
        self._i = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Completion:
        if callable(self._replies):
            return Completion(self._replies(messages))
        reply = self._replies[min(self._i, len(self._replies) - 1)]
        self._i += 1
        return Completion(reply)
