import asyncio
import backoff
from openai import OpenAI, AsyncOpenAI
from openai import APIError, RateLimitError, APIConnectionError, APITimeoutError
from typing import Any, Dict, List, Tuple, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import *
from utils.base import *
from utils.logger import *
from utils.utils import *

RETRYABLE_ERRORS = (APIError, RateLimitError, APIConnectionError, APITimeoutError)

@dataclass
class Response:
    text: str
    usage: Dict[str, int]
    hidden_states: torch.Tensor | None = None

def rstrip_iff_entire(s, subs):
  if s.endswith(subs):
    # If s ends with subs, return the string without the length of subs at the end
    return s[:-len(subs)]
  else:
    # Otherwise, return the original string
    return s

# TODO make it robust such that one of the particle dead (e.g. due to max tokens), the whole generation is not stopped
# TODO change stop_token to be a function called is_stopped
class StepGeneration:
    def __init__(
        self, 
        lm: AbstractLanguageModel,
        step_token: Union[str, List[str]], 
        max_steps: int, 
        stop_token: str = None, 
        temperature: float = 0.8, 
        include_stop_str_in_output: bool = False,  # If True, keep stop strings in output; if False, strip them 
        temperature_switch: Optional[Tuple[float, str, str]] = None, # (temperature, open_token, close_token)
    ):
        if not include_stop_str_in_output:
            assert isinstance(step_token, str), "step_token must be a string if include_stop_str_in_output is False"
        else:
            assert step_token is not None, "step_token must be provided if include_stop_str_in_output is True"
        self.lm = lm
        self.step_token = step_token
        self.max_steps = max_steps
        self.stop_token = stop_token
        self.temperature = temperature
        self.include_stop_str_in_output = include_stop_str_in_output
        self.temperature_switch = temperature_switch

    def set_system_prompt(self, system_prompt: str):
        self.lm.set_system_prompt(system_prompt)
    
    def set_stop_token(self, stop_token: str):
        self.stop_token = stop_token

    def _post_process(self, steps: List[str], stopped: bool = False) -> str:
        if self.include_stop_str_in_output:
            if stopped:
                last_step = steps[-1]
                last_step = rstrip_iff_entire(last_step, self.stop_token)
                steps = steps[:-1] + [last_step]
            return "".join(steps)
        else:
            response = self.step_token.join(steps)
            if not stopped:
                response += self.step_token
            return response
        
    def _get_temperatures(self, messages_list: Union[List[dict], List[List[dict]]]) -> List[float]:
        if self.temperature_switch is None:
            return [self.temperature]
        else:
            results = []
            for messages in messages_list:
                if messages[-1]["role"] == "assistant":
                    temperature, open_token, close_token = self.temperature_switch
                    if open_token in messages[-1].content and close_token not in messages[-1].content:
                        results.append(temperature)
                    else:
                        results.append(self.temperature)
                else:
                    results.append(self.temperature)

            return results
    
    async def generate(
        self,
        prompts: List[str], 
        steps_so_far: List[List[str]] = []
    ) -> List[Tuple[str, bool]]:
        messages_list = []
        for prompt, steps_so_far_per_prompt in zip(prompts, steps_so_far):
            messages = [
                {"role": "user", "content": prompt},
            ]
            if steps_so_far_per_prompt:
                messages.append({"role": "assistant", "content": self._post_process(steps_so_far_per_prompt)})
            messages_list.append(messages)
        # print("Messages list:", messages_list)
        next_steps = await self.lm.generate(
            messages_list, 
            stop=self.step_token, 
            temperatures=self._get_temperatures(messages_list), 
            # include_stop_str_in_output=self.include_stop_str_in_output
        )
        # print("Next steps:", next_steps)
        is_stopped = [len(steps_so_far_per_prompt) >= self.max_steps
                        for steps_so_far_per_prompt in steps_so_far]
        if self.stop_token:
            is_stopped = [is_stopped_per_prompt or self.stop_token in next_step.text
                            for is_stopped_per_prompt, next_step in zip(is_stopped, next_steps)]
        return list(zip(next_steps, is_stopped))



def default_on_backoff(details):
    logger.warning(f"Backing off {details['wait']:0.1f}s after {details['tries']} tries due to {details['exception']}")

class OpenAILanguageModel(AbstractLanguageModel):
    """wrapper for OpenAI chat/completion models"""
    def __init__(
        self,
        api_key: str,
        model_name: str,
        api_base: str | None = None,
        system_prompt: str | None = None,
        stop: str | None = None,
        is_chat: bool = True,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_tries: int = 8,
        max_concurrency: int = -1,
        replace_error_with_message: str | None = None,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_tries = max_tries
        self.max_concurrency = max_concurrency
        self.replace_error_with_message = replace_error_with_message

        self.stop = stop
        self.is_chat = is_chat
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True  # Important for models like Qwen
        )

        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.async_client = AsyncOpenAI(base_url=api_base, api_key=api_key)

    def _get_max_tokens_from_prompt(self, prompt: str, max_tokens: int, max_context: int = 4096, buffer: int = 32) -> int:
        """Compute max_tokens dynamically for an open-source LLM."""
        if max_tokens is None:
            return None
        prompt_tokens = len(self.tokenizer(prompt)["input_ids"])
        available_tokens = max_context - prompt_tokens - buffer
        return min(max_tokens, max(1, available_tokens))

    def _prepare_messages(self, messages):
        if self.is_chat:
            if self.system_prompt:
                return [{"role": "system", "content": self.system_prompt}] + messages
            else:
                return messages
        else:
            if self.system_prompt:
                return self.system_prompt + "\n\n".join([message["content"] for message in messages])
            else:
                return "\n\n".join([message["content"] for message in messages])
    
    @backoff.on_exception(
        backoff.expo,
        RETRYABLE_ERRORS,
        max_tries=8,
        on_backoff=default_on_backoff,
    )
    async def _text_completion(self, messages, stop=None, max_tokens=None, temperature=None, timeout=None):
        # print(self._prepare_messages(messages))
        prompt = self._prepare_messages(messages)
        response = await self.async_client.completions.create(
            model=self.model_name,
            prompt=prompt,
            stop=stop or self.stop,
            max_tokens=self._get_max_tokens_from_prompt(prompt, max_tokens or self.max_tokens),
            temperature=temperature if temperature is not None else self.temperature,
            timeout=timeout,
        )
        return response

    @backoff.on_exception(
        backoff.expo,
        RETRYABLE_ERRORS,
        max_tries=8,
        on_backoff=default_on_backoff,
    )
    async def _chat_completion(self, messages, stop, max_tokens, temperature, timeout):
        # print(self._prepare_messages(messages))
        return await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=self._prepare_messages(messages),
            stop=stop or self.stop,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            timeout=timeout,
        )

    async def generate(
        self, 
        messages_list, 
        stop=None, 
        max_tokens=None, 
        temperatures=None,
        timeout=None,
    ):
        semaphore = asyncio.Semaphore(
            len(messages_list) if self.max_concurrency == -1 else self.max_concurrency
        )

        async def safe_call(messages, temp, timeout):
            async with semaphore:
                try:
                    # print(messages, temp)
                    if self.is_chat:
                        response = await self._chat_completion(messages, stop, max_tokens, temp, timeout)
                        text = response.choices[0].message.content
                    else:
                        response = await self._text_completion(messages, stop, max_tokens, temp, timeout)
                        text = response.choices[0].text
                    return Response(text=text, usage=response.usage)
                except Exception as e:
                    logger.error(f"Error during async generation: {e}")
                    if self.replace_error_with_message is not None:
                        return self.replace_error_with_message
                    else:
                        raise

        temperatures = temperatures if temperatures and len(temperatures) == len(messages_list) \
                        else [self.temperature] * len(messages_list)
        tasks = [
            safe_call(messages, temp, timeout) 
            for messages, temp in zip(messages_list, temperatures, strict=True)
        ]

        return await asyncio.gather(*tasks)

    def evaluate(self, prompt: str, generation: str) -> list[float]:
        raise NotImplementedError("evaluate method not implemented")


class LocalLanguageModel(AbstractLanguageModel):
    def __init__(
        self,
        model_name_or_path: str,
        system_prompt: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        return_hidden_states: bool = False,
        max_tokens: int = 128,
        temperature: float = 0.7,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            output_hidden_states=return_hidden_states,
            torch_dtype="auto", #torch.float16 if "cuda" in device else torch.float32,
        ).to(device)
        self.device = device
        self.system_prompt = system_prompt
        self.return_hidden_states = return_hidden_states
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def _build_prompt(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # ensures assistant placeholder added
        )

    def generate(self, messages_list: list[list[dict]]) -> list[Response]:
        results = []
        for messages in messages_list:
            prompt = self._build_prompt(messages)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    output_hidden_states=self.return_hidden_states,
                    return_dict_in_generate=True,
                )

            generated_text = self.tokenizer.decode(
                outputs.sequences[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            hidden = outputs.hidden_states[-1] if self.return_hidden_states and outputs.hidden_states else None
            usage = {
                "prompt_tokens": inputs["input_ids"].shape[-1],
                "completion_tokens": outputs.sequences.shape[-1] - inputs["input_ids"].shape[-1],
            }

            results.append(Response(text=generated_text, usage=usage, hidden_states=hidden))

        return results
    
    def evaluate(self, prompt: str, generation: str) -> list[float]:
        raise NotImplementedError("evaluate method not implemented")