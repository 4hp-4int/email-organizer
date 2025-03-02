from typing import List
from bertopic.representation import TextGeneration
from hdbscan import HDBSCAN
from umap import UMAP

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from torch import bfloat16
import transformers
from xconfig import Config

config = Config()


class LlamaAgent:
    def __init__(
        self, model_path: str = None, model_id="meta-llama/Llama-2-7b-chat-hf"
    ):
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

        self.llama2 = TextGeneration(
            self.generator, prompt=self.prompt, doc_length=100, tokenizer=self.tokenizer
        )

    def _load_prompt_from_file(self, filename: str) -> str:
        """Loads a prompt from a plaintext file."""
        try:
            with open(filename, "r") as f:
                return f.read().strip()
        except FileNotFoundError as e:
            print(f"Error: Prompt file '{filename}' not found.")
            raise e

    def reduce_embeddings(
        self, np_embeddings, neighbors=15, components=2, min_dist=0.0
    ):
        # Reduce the embeddings for visualization
        reduced_embeddings = UMAP(
            n_neighbors=neighbors,
            n_components=components,
            min_dist=min_dist,
            metric="cosine",
            random_state=42,
        ).fit_transform(np_embeddings)

        return reduced_embeddings

    @classmethod
    def parse_labels(cls, topics: List[int]):
        llama2_labels = [
            label[0][0].split("\n")[0]  # Get first line of the label
            for label in topics.values()
            if label
            and isinstance(label, list)
            and label[0]
            and isinstance(label[0], tuple)
            and label[0][0].strip()
        ]
        return llama2_labels
