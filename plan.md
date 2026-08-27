# Product Requirements Document (PRD): Local Multimodal Windows Assistant ("Jarvis")

## 1. Product Overview
**Objective:** Build a local, multimodal desktop assistant powered by the Gemma 4 E4B instruction-tuned model. The agent will run autonomously on Windows, listening for the wake word "Jarvis," analyzing screen captures, and executing local Windows OS commands (PowerShell/cmd) via function calling, entirely without cloud API dependencies.

**Target Hardware:** NVIDIA GeForce RTX 4060 (8GB VRAM).
**Target OS:** Windows 10 / Windows 11.
**Target User:** AI Engineer / Developer needing a hands-free local Windows assistant to review code, search the web, take notes, and execute local scripts.

## 2. Technical Stack & Hardware Optimization
*   **Core LLM:** `google/gemma-4-E4B-it` (features native text, image, and audio encoders).
*   **Hardware Acceleration:** NVIDIA CUDA (Compute Capability 8.9 / Ada Lovelace).
*   **Quantization / Precision:** `bitsandbytes` 4-bit (NF4) or 8-bit precision, or `llama.cpp` CUDA backend, ensuring the model weights, KV cache, audio, and visual embeddings stay comfortably within the 8GB VRAM budget of the RTX 4060.
*   **Inference & Backend:** `PyTorch` (CUDA enabled) and Hugging Face `transformers` with `accelerate`.
*   **Agent Orchestration:** `LangGraph` (for managing the stateful, multi-step agent execution loop and tool calling sequences).
*   **Wake Word Engine:** `openWakeWord` (running via `onnxruntime-gpu` or CPU on Windows) utilizing the pre-trained `"hey_jarvis"` model.
*   **Audio Capture:** `sounddevice` configured for Windows WASAPI/DirectSound to capture microphone input, along with `numpy` (formatting 16kHz float32 audio arrays).
*   **Screen Capture:** `mss` (fast, cross-platform screenshot utility) and `Pillow` (PIL) for image formatting.
*   **OS Automation:** Standard Python `subprocess` and `os` libraries for executing Windows functions.
*   **Language:** Python 3.10+

## 3. Architecture & Data Flow

The system operates in a continuous "Listen -> Perceive -> Plan -> Act" loop, modeled as a compiled state graph in LangGraph.

### Core Modules
1.  **Wake Word Listener (Background):** `sounddevice` continuously feeds short audio chunks into `openWakeWord`. Once the threshold for "hey jarvis" is triggered, it initiates the main recording loop.
2.  **Command Capture (Audio):** Records up to 15-30 seconds of user audio following the wake word.
3.  **Context Gatherer (Vision):** The system instantly captures a screenshot of the active Windows display using `mss`, dynamically resizing/downscaling to keep visual token memory minimal on the RTX 4060.
4.  **Prompt Construction:** The state payload is assembled using Gemma 4's multimodal prompt structure: `<image_tensor> + <text_system_prompt> + <audio_tensor>`.
5.  **Inference & Planning:** The model runs inference on the RTX 4060 via CUDA within the LangGraph node and outputs a JSON-formatted reasoning block followed by an action command.
6.  **Execution Engine:** The graph transitions to a Tool Execution node if a function is called. It executes the local Python function and appends the result back into the agent's message state for a follow-up response.

## 4. Windows-Specific Tool Definitions

The model must be provided with a strict JSON schema for the tools it can execute. These specific tools should be implemented in a `tools.py` module:

| Tool Name | Parameters | Windows Implementation Details |
| :--- | :--- | :--- |
| `execute_terminal_command` | `command` (string) | Runs a command using `subprocess.run(..., shell=True)` on Windows. Must return `stdout`/`stderr` back to the agent context. |
| `read_file_content` | `file_path` (string) | Reads the contents of a local file (handling Windows `\\` pathing). |
| `create_note` | `content` (str), `title` (str, optional) | Writes a `.txt` file directly to `os.path.join(os.path.expanduser("~"), "Desktop")` or opens it via `subprocess.Popen(["notepad.exe", file_path])`. |
| `web_search` | `query` (str), `open_in_browser` (bool) | If `True`, uses `webbrowser.open()` to launch the default Windows browser (Edge/Chrome). If `False`, uses `ddgs` to fetch text snippets silently; empty searches return `No search results found`. |

## 5. Implementation Milestones

When generating the codebase, proceed strictly in this order to ensure modular stability on Windows:

### Phase 1: Environment Setup & RTX 4060 VRAM Allocation
*   **Task 1.1:** Setup the Python environment with PyTorch CUDA (`torch.cuda.is_available() == True`). Load `google/gemma-4-E4B-it` using 4-bit `BitsAndBytesConfig` (or 8-bit) to ensure peak VRAM usage remains strictly under 8GB.
*   **Task 1.2:** Implement `wake_word.py`. Initialize the `"hey_jarvis"` model. Configure `sounddevice` to stream microphone audio (using `channels=1`, `samplerate=16000`) into the model's prediction loop.
*   **Task 1.3:** Once "Jarvis" is detected, trigger the main `audio_capture.py` function to record the user command. The final command audio must be formatted as mono-channel, 16 kHz float32 waveforms in the range `[-1, 1]`.

### Phase 2: Multimodal Input Assembly & Prompting
*   **Task 2.1:** Create `agent.py`. Define the system prompt instructing the agent on its role as a Windows OS assistant named Jarvis.
*   **Task 2.2:** Implement `screen_capture.py` using `mss`. Output must be a PIL Image object resized to fit within Gemma's vision token limits.
*   **Task 2.3:** Implement the multimodal input pipeline. Feed the PIL image, the audio array, and the system prompt into the tokenizer/processor and push tensors directly to `cuda:0`.

### Phase 3: LangGraph Integration & Tool Execution
*   **Task 3.1:** Implement `tools.py` containing the python wrappers for the tools defined in Section 4. Ensure paths use `os.path` for Windows compatibility.
*   **Task 3.2:** Build the recursive execution loop using LangGraph. Define nodes for inference and tool execution. If the model outputs a tool call, transition to the tool node, execute the function, append the result to the state history, and pass it back to the model to generate the final response.

## 6. Success Criteria & Edge Cases
*   **VRAM Management:** Peak memory consumption during concurrent vision and audio processing must not exceed 7.5GB to avoid CUDA out-of-memory (OOM) crashes on the RTX 4060.
*   **Audio Driver Handling:** Ensure `sounddevice` fails gracefully and prompts the user to select the correct Windows microphone index if WASAPI/DirectSound fails.
*   **Safety/Execution Limits:** The system must implement a confirmation loop (e.g., `input("Approve execution of [command]? Y/N")`) before the agent runs any `execute_terminal_command` outputs to prevent accidental destructive actions on the Windows host.