# SPDX-License-Identifier: Apache-2.0
"""
MLX Language Model wrapper.

This module provides a wrapper around mlx-lm for LLM inference,
integrating with vLLM's model execution system.
"""

import logging
from dataclasses import dataclass
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """Output from text generation."""

    text: str
    tokens: list[int]
    finish_reason: str | None = None


@dataclass
class StreamingOutput:
    """Streaming output chunk."""

    text: str
    token: int
    finished: bool = False
    finish_reason: str | None = None
    prompt_tokens: int = 0


class MLXLanguageModel:
    """
    Wrapper around mlx-lm for LLM inference.

    This class provides a unified interface for loading and running
    inference on language models using Apple's MLX framework.

    Example:
        >>> model = MLXLanguageModel("mlx-community/Llama-3.2-3B-Instruct-4bit")
        >>> output = model.generate("Hello, how are you?", max_tokens=100)
        >>> print(output.text)
    """

    def __init__(
        self,
        model_name: str,
        tokenizer_name: str | None = None,
        trust_remote_code: bool = False,
        mtp: bool = False,
    ):
        """
        Initialize the MLX language model.

        Args:
            model_name: HuggingFace model name or local path
            tokenizer_name: Optional separate tokenizer name
            trust_remote_code: Whether to trust remote code
            mtp: Enable native MTP speculative decoding (model must have MTP head)
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.trust_remote_code = trust_remote_code
        self._mtp = mtp

        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load the model and tokenizer."""
        if self._loaded:
            return

        try:
            from ..utils.tokenizer import load_model_with_fallback

            logger.info(f"Loading model: {self.model_name}")

            # Build tokenizer config
            tokenizer_config = {"trust_remote_code": self.trust_remote_code}

            # Qwen3 fix: eos_token changed from <|im_end|> to <|endoftext|>
            # but chat template still uses <|im_end|>, so we need to set it explicitly
            if "qwen3" in self.model_name.lower() or "Qwen3" in self.model_name:
                tokenizer_config["eos_token"] = "<|im_end|>"
                logger.info("Qwen3 detected: setting eos_token to <|im_end|>")

            self.model, self.tokenizer = load_model_with_fallback(
                self.model_name,
                tokenizer_config=tokenizer_config,
            )

            self._loaded = True
            logger.info(f"Model loaded successfully: {self.model_name}")

        except ImportError:
            raise ImportError(
                "mlx-lm is required for LLM inference. "
                "Install with: pip install mlx-lm"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _create_sampler(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
    ):
        """Create a sampler for text generation."""
        from mlx_lm.sample_utils import make_sampler

        return make_sampler(
            temp=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )

    def _create_logits_processors(
        self,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        repetition_penalty: float | None = None,
    ):
        """Create logits processors for penalty parameters."""
        from mlx_lm.sample_utils import make_logits_processors

        processors = make_logits_processors(
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
        )
        return processors if processors else None

    # Token ID for </think> — used to detect thinking completion.
    _think_close_token_id: int | None = None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        repetition_penalty: float | None = None,
        stop: list[str] | None = None,
    ) -> GenerationOutput:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeating tokens
            stop: List of stop sequences

        Returns:
            GenerationOutput with generated text and tokens
        """
        if not self._loaded:
            self.load()

        from mlx_lm import stream_generate

        # Resolve </think> token ID once
        if self._think_close_token_id is None:
            try:
                ids = self.tokenizer.encode("</think>", add_special_tokens=False)
                self._think_close_token_id = ids[0] if len(ids) == 1 else -1
            except Exception:
                self._think_close_token_id = -1

        # Create sampler with parameters
        sampler = self._create_sampler(temperature, top_p, top_k, min_p)

        # Create logits processors for penalties
        logits_processors = self._create_logits_processors(
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
        )

        gen_kwargs = {"sampler": sampler}
        if logits_processors:
            gen_kwargs["logits_processors"] = logits_processors

        # Determine whether the prompt ends inside a <think> block.
        prompt_has_think = prompt.rstrip().endswith("<think>")
        thinking_budget = max_tokens // 2 if prompt_has_think else max_tokens

        accumulated_text = ""
        tokens: list[int] = []
        finish_reason: str | None = None
        think_close_seen = False

        for resp in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            **gen_kwargs,
        ):
            accumulated_text += resp.text
            tokens.append(resp.token)

            if resp.token == self._think_close_token_id:
                think_close_seen = True

            if (
                prompt_has_think
                and not think_close_seen
                and len(tokens) >= thinking_budget
            ):
                logger.info(
                    "Thinking budget (%d tokens) exceeded without </think> — "
                    "force-closing thinking block.",
                    thinking_budget,
                )
                accumulated_text += "</think>\n"
                finish_reason = "thinking_budget"
                break

            if resp.finish_reason is not None:
                finish_reason = resp.finish_reason
                break

        # If we force-closed thinking, run a second pass for the answer.
        if finish_reason == "thinking_budget":
            answer_budget = max_tokens - len(tokens)
            if answer_budget > 0:
                answer_prompt = prompt + accumulated_text
                for resp in stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt=answer_prompt,
                    max_tokens=answer_budget,
                    **gen_kwargs,
                ):
                    accumulated_text += resp.text
                    tokens.append(resp.token)
                    if resp.finish_reason is not None:
                        finish_reason = resp.finish_reason
                        break

            if finish_reason == "thinking_budget":
                finish_reason = "length"

        if finish_reason is None:
            finish_reason = "length" if len(tokens) >= max_tokens else "stop"

        return GenerationOutput(
            text=accumulated_text,
            tokens=tokens,
            finish_reason=finish_reason,
        )

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        repetition_penalty: float | None = None,
        stop: list[str] | None = None,
    ) -> Iterator[StreamingOutput]:
        """
        Stream text generation token by token.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeating tokens
            stop: List of stop sequences

        Yields:
            StreamingOutput for each generated token
        """
        if not self._loaded:
            self.load()

        from mlx_lm import stream_generate

        # Create sampler with parameters
        sampler = self._create_sampler(temperature, top_p)

        # Count prompt tokens once upfront
        num_prompt_tokens = len(self.tokenizer.encode(prompt))

        token_count = 0
        accumulated_text = ""

        mtp_kwargs = {}
        if self._mtp:
            mtp_kwargs["mtp"] = True

        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            **mtp_kwargs,
        ):
            token_count += 1
            # response.text is the new token text (not accumulated)
            new_text = response.text
            accumulated_text += new_text

            # Check for stop sequences
            should_stop = False
            if stop:
                for stop_seq in stop:
                    if stop_seq in accumulated_text:
                        should_stop = True
                        break

            finished = should_stop or token_count >= max_tokens
            finish_reason = None
            if finished:
                finish_reason = "stop" if should_stop else "length"

            yield StreamingOutput(
                text=new_text,
                token=response.token if hasattr(response, "token") else 0,
                finished=finished,
                finish_reason=finish_reason,
                prompt_tokens=num_prompt_tokens,
            )

            if finished:
                break

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate a chat response.

        Args:
            messages: List of chat messages [{"role": "user", "content": "..."}]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            tools: Optional list of tools for function calling
            **kwargs: Additional generation parameters

        Returns:
            GenerationOutput with the assistant's response
        """
        if not self._loaded:
            self.load()

        # Apply chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            # Enable thinking mode for non-coder models
            enable_thinking = "coder" not in self.model_name.lower()

            # Build kwargs for apply_chat_template
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": enable_thinking,
            }

            # Add tools if provided and supported
            if tools:
                template_kwargs["tools"] = tools

            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    **template_kwargs,
                )
            except TypeError:
                # Some templates don't support all kwargs
                for key in ["tools", "enable_thinking"]:
                    if key in template_kwargs:
                        del template_kwargs[key]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    **template_kwargs,
                )
        else:
            # Fallback: simple concatenation
            prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
            prompt += "\nassistant:"

        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._loaded:
            return {"loaded": False, "model_name": self.model_name}

        info = {
            "loaded": True,
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
        }

        # Try to get model config
        if hasattr(self.model, "config"):
            config = self.model.config
            info.update(
                {
                    "vocab_size": getattr(config, "vocab_size", None),
                    "hidden_size": getattr(config, "hidden_size", None),
                    "num_layers": getattr(config, "num_hidden_layers", None),
                    "num_heads": getattr(config, "num_attention_heads", None),
                }
            )

        return info

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"<MLXLanguageModel model={self.model_name} status={status}>"
