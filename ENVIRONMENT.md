Below is a more in-depth how-to guide derived from your Bash history, showing how you set up your environment, installed dependencies, compiled llama.cpp, downloaded Qwen, and worked with your Python code. These instructions combine the key commands you used. Feel free to adjust details (versions, user names, directories) based on your specific system setup.

---


---

# 1. Install & Configure Poetry

1. **Install Poetry via pipx**:
   ```bash
   pipx install poetry
   pipx ensurepath
   # or if you prefer the official script:
   curl -sSL https://install.python-poetry.org | python3 -
   ```
2. **Create/Activate a Poetry-managed virtualenv**:
   ```bash
   cd /path/to/your/project
   poetry install
   poetry shell
   ```
3. **Check your Python path**:
   ```bash
   which python3
   python3 --version
   ```

---

# 2. Python Dependencies & Project Setup

In your project, you frequently used:

1. **Installing direct packages with Poetry**:
   ```bash
   poetry add bertopic
   poetry add spacy
   poetry add sentence-transformers
   poetry add jupyterlab
   poetry add loguru
   ...
   ```
2. **Installing via pip** (if Poetry had conflicts):
   ```bash
   pip install bertopic
   ```

3. **Install SpaCy English model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Generate or update your `requirements.txt`** if you need a pip-based install:
   ```bash
   pip freeze > requirements.txt
   ```

---

# 3. GPU / CUDA Setup

From your history, you installed or updated CUDA 12.8. Steps:

1. **Add the NVIDIA package repository**:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   ```
2. **Install CUDA toolkit**:
   ```bash
   sudo apt-get -y install cuda-toolkit-12-8
   ```
   Make sure your PATH includes `/usr/local/cuda/bin`:
   ```bash
   echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   nvcc --version
   ```

---

# 4. llama-cpp-python (GPU Build)

You compiled **llama-cpp-python** with GPU support. The key environment variables:

```bash
CMAKE_ARGS="-DCUDAToolkit_ROOT=/usr/local/cuda -DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

- `-DGGML_CUDA=on` enables GPU acceleration in the underlying GGML code.
- `CUDAToolkit_ROOT=/usr/local/cuda` points cmake to the installed CUDA toolkit.

If you want cublas-based acceleration:

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

---

# 5. Qwen Model (Alibaba Cloud’s LLM)

You downloaded Qwen 2.5 using the Hugging Face CLI with the `.gguf` format:

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
  qwen2.5-7b-instruct-q5_k.gguf --local-dir .
```

That places the model weights (`qwen2.5-7b-instruct-q5_k.gguf`) into your current directory. You then can load it via `llama-cli` or other GGUF-compatible tools.

---

# 6. Building llama.cpp (Optional, if you want a CLI instead of Python)

You also compiled the official [llama.cpp](https://github.com/ggerganov/llama.cpp) so you can run `llama-cli`. Here’s the minimal approach:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release

# Then you can move the llama-cli to a system path
sudo mv build/bin/llama-cli /usr/local/bin/
```

Now you can run:

```bash
llama-cli -m qwen2.5-7b-instruct-q2_k.gguf \
  -p "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." \
  -n 512
```

---

# 7. Running Your Email Organizer

From your code structure:

- `app.py` has Typer commands:
  - `collect_and_store`  
  - `create_inbox_folders`  
  - `organize` (or `review_and_categorize`)  
  - `train_topic_model`  
  - `train_classifier_model_v2`  

To run them:

```bash
python3 app.py collect_and_store
python3 app.py create_inbox_folders
python3 app.py organize
python3 app.py train_topic_model
# etc.
```

If you’re using Poetry:
```bash
poetry run python app.py collect_and_store
```
---

## Putting It All Together

1. **Install system packages** (for building, CUDA, dev libs).
2. **Install Poetry** (or use pip in a virtualenv).
3. **Install project requirements**:
   - `poetry install` or `pip install -r requirements.txt`.
4. **Install additional LLMs** as needed:
   - `huggingface-cli download ...` for Qwen.
5. **Optional**: Build `llama-cpp-python` with GPU support.
6. **Optional**: Build `llama.cpp` from source for a command-line LLM runner.
7. **Run your email organizing commands** in `app.py`.