
from __future__ import annotations

from pathlib import Path
import threading
import time
from typing import Any, Sequence

from qprobe.backends import (
    GenerationRequest,
    GenerationResponse,
    NLIRequest,
    NLIResponse,
    NLILabel,
    TokenUsage,
)
from qprobe.errors import ConfigurationError, ModelCallError


def _load_transformers() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as error:
        raise ConfigurationError(
            "Local model adapters require torch and transformers"
        ) from error
    return torch, transformers


def _require_model_directory(path: Path) -> None:
    if not isinstance(path, Path) or not path.is_dir():
        raise ConfigurationError("model_path must be an existing directory")


def _synchronize_cuda(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class TransformersGenerationBackend:

    def __init__(
        self,
        *,
        model_id: str,
        model_path: Path,
        device: str = "cuda:0",
        dtype: str = "auto",
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ConfigurationError("model_id must be non-empty")
        _require_model_directory(model_path)
        if dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ConfigurationError("Unsupported local generation dtype")
        torch, transformers = _load_transformers()
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise ConfigurationError("CUDA was requested but is unavailable")
        resolved_dtype = (
            "auto" if dtype == "auto" else getattr(torch, dtype)
        )
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True,
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                str(model_path),
                dtype=resolved_dtype,
                device_map={"": str(resolved_device)},
                local_files_only=True,
            )
        except Exception as error:
            raise ModelCallError("Cannot load the controlled target checkpoint") from error
        model.eval()
        self._model_id = model_id
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = resolved_device
        self._lock = threading.Lock()

    def generate(self, request: GenerationRequest) -> GenerationResponse:

        if not isinstance(request, GenerationRequest):
            raise ConfigurationError("request must be a GenerationRequest")
        if request.model_id != self._model_id:
            raise ModelCallError("Generation request uses an unexpected model_id")
        try:
            tokenized = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": request.prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            input_ids = getattr(tokenized, "input_ids", tokenized).to(self._device)
            attention_mask = getattr(tokenized, "attention_mask", None)
            if attention_mask is None:
                attention_mask = self._torch.ones_like(input_ids)
            else:
                attention_mask = attention_mask.to(self._device)
        except Exception as error:
            raise ModelCallError("Cannot apply the target model chat template") from error
        generation_arguments: dict[str, object] = {
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.temperature > 0,
            "pad_token_id": (
                self._tokenizer.pad_token_id
                if self._tokenizer.pad_token_id is not None
                else self._tokenizer.eos_token_id
            ),
        }
        if request.temperature > 0:
            generation_arguments["temperature"] = request.temperature

        with self._lock:
            try:
                with self._torch.random.fork_rng():
                    self._torch.manual_seed(request.seed)
                    if self._device.type == "cuda":
                        self._torch.cuda.manual_seed_all(request.seed)
                    _synchronize_cuda(self._torch, self._device)
                    started = time.perf_counter()
                    with self._torch.inference_mode():
                        output_ids = self._model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            **generation_arguments,
                        )
                    _synchronize_cuda(self._torch, self._device)
                    latency_seconds = time.perf_counter() - started
            except Exception as error:
                raise ModelCallError("Target generation failed") from error

        generated_ids = output_ids[0, input_ids.shape[-1] :]
        text = self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        if not text:
            raise ModelCallError("Target model returned an empty generation")
        eos_ids = self._model.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_set = {eos_ids}
        else:
            eos_set = set(eos_ids or ())
        finish_reason = (
            "stop"
            if generated_ids.numel() and int(generated_ids[-1]) in eos_set
            else "length"
        )
        return GenerationResponse(
            text=text,
            model_id=self._model_id,
            finish_reason=finish_reason,
            usage=TokenUsage(
                input_tokens=int(input_ids.numel()),
                output_tokens=int(generated_ids.numel()),
            ),
            latency_seconds=latency_seconds,
        )

    def generate_many(
        self, requests: Sequence[GenerationRequest]
    ) -> tuple[GenerationResponse, ...]:

        return tuple(self.generate(request) for request in requests)


class TransformersNLIBackend:

    def __init__(
        self,
        *,
        model_id: str,
        model_path: Path,
        device: str = "cuda:0",
        dtype: str = "auto",
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ConfigurationError("model_id must be non-empty")
        _require_model_directory(model_path)
        if dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ConfigurationError("Unsupported local NLI dtype")
        torch, transformers = _load_transformers()
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise ConfigurationError("CUDA was requested but is unavailable")
        resolved_dtype = (
            "auto" if dtype == "auto" else getattr(torch, dtype)
        )
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True,
            )
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                str(model_path),
                dtype=resolved_dtype,
                device_map={"": str(resolved_device)},
                local_files_only=True,
            )
        except Exception as error:
            raise ModelCallError("Cannot load the controlled NLI checkpoint") from error
        label_ids = {
            str(label).casefold(): int(index)
            for index, label in model.config.id2label.items()
        }
        expected = {label.value for label in NLILabel}
        if set(label_ids) != expected:
            raise ModelCallError("NLI checkpoint labels are incompatible")
        model.eval()
        self._model_id = model_id
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = resolved_device
        self._label_ids = {
            label: label_ids[label.value] for label in NLILabel
        }

    def classify(self, request: NLIRequest) -> NLIResponse:

        if not isinstance(request, NLIRequest):
            raise ConfigurationError("request must be an NLIRequest")
        if request.model_id != self._model_id:
            raise ModelCallError("NLI request uses an unexpected model_id")
        try:
            inputs = self._tokenizer(
                request.premise,
                request.hypothesis,
                return_tensors="pt",
                truncation=True,
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            _synchronize_cuda(self._torch, self._device)
            started = time.perf_counter()
            with self._torch.inference_mode():
                logits = self._model(**inputs).logits[0]
                probabilities = self._torch.softmax(logits.float(), dim=-1)
            _synchronize_cuda(self._torch, self._device)
            latency_seconds = time.perf_counter() - started
        except Exception as error:
            raise ModelCallError("NLI classification failed") from error
        scores = {
            label: float(probabilities[index].item())
            for label, index in self._label_ids.items()
        }
        selected = max(NLILabel, key=scores.__getitem__)
        return NLIResponse(
            label=selected,
            scores=scores,
            model_id=self._model_id,
            latency_seconds=latency_seconds,
        )

    def classify_many(
        self, requests: Sequence[NLIRequest]
    ) -> tuple[NLIResponse, ...]:

        return tuple(self.classify(request) for request in requests)
