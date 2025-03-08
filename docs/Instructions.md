

# Install Llama.CPP
sudo apt install cmake
git clone https://github.com/ggml-org/llama.cpp
sudo apt-get install cuda-toolkit
cmake -B build -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build build --config Release




1. **Download the model from Hugging Face:**

   ```bash
   huggingface-cli download Qwen/Qwen2.5-14B-Instruct-1M --local-dir .
   ```

2. **Convert the downloaded model to GGUF format using the llama.cpp conversion script:**

   ```bash
   python3 ~/tools/llama.cpp/convert_hf_to_gguf.py email-organizer --outfile Qwen2.5-14b-Instruct-1M.gguf
   ```

3. **Quantize the GGUF model with llama-quantize:**

   ```bash
   llama-quantize Qwen2.5-14b-Instruct-1M.gguf  Qwen2.5-14b-Instruct-1M-q8_0.gguf Q8_0
   ```