from typing import List
from bertopic.representation import TextGeneration
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from torch import bfloat16
import transformers
from xconfig import Config

config = Config()


class LlamaAgent:
    def __init__(self, model_id="meta-llama/Llama-2-7b-chat-hf"):
        self.model_id = model_id
        self.bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=self.bnb_config,
            device_map="auto",
        )
        self.model.eval()
        self.generator = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            temperature=0.1,
            max_new_tokens=config.LLAMA_MAX_NEW_TOKENS,
            repetition_penalty=1.1,
        )
        # Load prompts from files
        self._system_prompt = self._load_prompt_from_file(
            "prompts/llama2_system_prompt.txt"
        )
        self._example_prompt = self._load_prompt_from_file(
            "prompts/llama2_example_prompt.txt"
        )
        self._main_prompt = self._load_prompt_from_file(
            "prompts/llama2_main_prompt.txt"
        )

        # Set up llama2's prompt
        self.prompt = self._system_prompt + self._example_prompt + self._main_prompt

        self.llama2 = TextGeneration(self.generator, prompt=self.prompt)

    def _load_prompt_from_file(self, filename: str) -> str:
        """Loads a prompt from a plaintext file."""
        try:
            with open(filename, "r") as f:
                return f.read().strip()
        except FileNotFoundError as e:
            print(f"Error: Prompt file '{filename}' not found.")
            raise e
