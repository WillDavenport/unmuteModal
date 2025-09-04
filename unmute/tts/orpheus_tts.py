"""
Orpheus TTS Implementation for Modal
Based on Baseten Truss examples for optimal performance
"""

from typing import Any, Iterator, List, Awaitable, AsyncIterator
from transformers import AutoTokenizer
import torch
import fastapi
from snac import SNAC
from pathlib import Path
import numpy as np
from fastapi.responses import StreamingResponse, Response
import batched
import re
import time
import uuid
import asyncio
import threading
import logging

# force inference mode during the lifetime of the script
_inference_mode_raii_guard = torch._C._InferenceMode(True)

_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")
SNAC_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Max batch kept high for throughput; we reduce batching wait below for low latency
SNAC_MAX_BATCH = 64
PREPROCESS_STREAM = torch.Stream(SNAC_DEVICE)
MAX_CHARACTERS_INPUT = 6144

logger = logging.getLogger(__name__)


class SnacModelBatched:
    def __init__(self):
        self.dtype_decoder = torch.float32
        compile_background = False
        use_compile = True
        model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        model = model.to(SNAC_DEVICE)

        model.decoder = model.decoder.to(self.dtype_decoder)

        self.snac_model = model
        self.stream = torch.Stream()
        if use_compile:
            if compile_background:
                # Compile the model in a separate thread, experimental.
                threading.Thread(target=self.compile, daemon=True).start()
            else:
                # Compile the model in the main thread
                self.compile()

    def compile(self):
        model = self.snac_model
        # Compile the model with torch.compile
        decoder = torch.compile(model.decoder, dynamic=True)
        quantizer = torch.compile(model.quantizer, dynamic=True)
        t = time.time()
        logging.info("starting torch.compile")
        for bs_size in range(1, max(SNAC_MAX_BATCH, 1)):
            codes = [
                torch.randint(1, 4096, (bs_size, 4)).to(SNAC_DEVICE),
                torch.randint(1, 4096, (bs_size, 8)).to(SNAC_DEVICE),
                torch.randint(1, 4096, (bs_size, 16)).to(SNAC_DEVICE),
            ]
            with torch.inference_mode():
                intermed = quantizer.from_codes(codes)
                decoder(intermed.to(self.dtype_decoder))
        logging.info(f"torch.compile took {time.time() - t:.2f} seconds")
        self.snac_model.decoder = decoder
        self.snac_model.quantizer = quantizer

    # Lower timeout to reduce per-chunk latency when serving a single stream
    @batched.dynamically(batch_size=SNAC_MAX_BATCH, timeout_ms=2)
    def batch_snac_model(
        self, items: list[dict[str, list[torch.Tensor]]]
    ) -> list[torch.Tensor]:
        # Custom processing logic here
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            all_codes = [codes["codes"] for codes in items]
            can_be_batched = len(items) > 1 and all(
                codes[0].shape == all_codes[0][0].shape for codes in all_codes
            )
            if can_be_batched:
                # stacked_codes = [(b,4), (b,8), (b,16)]
                stacked_codes: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = [
                    torch.cat(  # codes is list[torch.Tensor]
                        [item[i] for item in all_codes], dim=0
                    )
                    for i in range(3)
                ]
                stacked_z_q = self.snac_model.quantizer.from_codes(stacked_codes)
                output_batched = self.snac_model.decoder(
                    stacked_z_q.to(self.dtype_decoder)
                )[:, :, 2048:4096].to(torch.float32)

                out = output_batched.split(
                    1, dim=0
                )  # unbatch the output into len(items) tensors of shape (1, 1, x)
            else:
                # items can't be batched
                if len(items) > 1:
                    # items can't cant be concatenated (no padding)
                    logging.warning(
                        "Warning: items can't be batched, using individual decoding."
                    )
                # if we have a single item, we need to do the same thing as above
                # but without concatenating
                out: list[torch.Tensor] = []
                for codes in all_codes:
                    stacked_z_q = self.snac_model.quantizer.from_codes(codes)
                    out.append(
                        self.snac_model.decoder(stacked_z_q.to(self.dtype_decoder))[
                            :, :, 2048:4096
                        ].to(torch.float32)
                    )
            self.stream.synchronize()  # make sure the results are ready
            return out


def turn_token_into_id(token_string: int, index: int):
    """Extract and convert the last custom token ID from a string."""
    return token_string - 10 - ((index % 7) * 4096)


def split_custom_tokens(s: str) -> List[int]:
    """
    Extracts all substrings enclosed in <custom_token_…> from the input string.
    """
    matches = _TOKEN_RE.findall(s)
    return [int(match) for match in matches if match != "0"]


async def tokens_decoder(
    token_gen: AsyncIterator, request_id: str, start_time: int
) -> AsyncIterator[bytes]:
    """Decoder that pipelines convert_to_audio calls but enforces strict in-order yields."""
    assert hasattr(token_gen, "__aiter__")
    audio_queue = asyncio.Queue()

    async def producer(token_gen: AsyncIterator):
        buffer: list[int] = []
        count = 0
        tft = 0
        async for token_sim in token_gen:
            if tft == 0:
                tft = time.time()
            for tok_str in split_custom_tokens(token_sim):
                token = turn_token_into_id(int(tok_str), count)
                buffer.append(token)
                count += 1
                # every 7 tokens → one frame; once we have at least 28 tokens, we extract the last 28
                if count % 7 == 0 and count > 27:
                    buf_to_proc = buffer[-28:]
                    task = asyncio.create_task(convert_to_audio(buf_to_proc))
                    audio_queue.put_nowait(task)
        audio_queue.put_nowait(None)
        elapsed = time.time() - start_time
        time_to_first_token = tft - start_time
        time_of_generation = time.time() - tft
        token_generation_speed = count / time_of_generation
        logging.info(
            f"Finished `{request_id}`, total tokens : {count}, time: {elapsed:.2f}s. "
            f"tokens/s generation: {token_generation_speed:.2f} (ttft: {time_to_first_token:.2f}s, generation time: {time_of_generation:.2f}s)"
            f" real-time factor once streaming started: {(token_generation_speed / 100):.2f} "
        )

    producer_task = asyncio.create_task(producer(token_gen))

    while True:
        # wait for the next audio conversion to finish
        task: None | Awaitable[bytes | None] = await audio_queue.get()
        if task is None:
            break
        audio_bytes = await task
        if audio_bytes is not None:
            yield audio_bytes
        audio_queue.task_done()
    assert audio_queue.empty(), (
        f"audio queue is not empty: e.g. {audio_queue.get_nowait()}"
    )
    await producer_task


@torch.inference_mode()
async def convert_to_audio(frame_ids: list[int]) -> bytes | None:
    """Convert a list of token IDs into audio bytes efficiently.

    frame_ids:
    - list of token IDS (phonemes) of length 28 or less.
    - 7 tokens = 1 frame
    """
    n = len(frame_ids) // 7
    if n == 0:
        return None

    arr = torch.tensor(frame_ids[: n * 7], dtype=torch.int32, device="cpu")
    if torch.cuda.is_available():
        try:
            arr = arr.pin_memory()
        except Exception:
            pass
    mat = arr.view(n, 7)
    codes_0 = mat[:, 0]
    codes_1 = mat[:, [1, 4]].reshape(-1)
    codes_2 = mat[:, [2, 3, 5, 6]].reshape(-1)
    if (
        ((codes_0 < 0) | (codes_0 > 4096)).any()
        or ((codes_1 < 0) | (codes_1 > 4096)).any()
        or ((codes_2 < 0) | (codes_2 > 4096)).any()
    ):
        logging.warning("Warn: Invalid token IDs detected, skipping audio generation.")
        return None
    with torch.cuda.stream(PREPROCESS_STREAM):
        codes = [
            codes_0.unsqueeze(0).to(SNAC_DEVICE, non_blocking=True),
            codes_1.unsqueeze(0).to(SNAC_DEVICE, non_blocking=True),
            codes_2.unsqueeze(0).to(SNAC_DEVICE, non_blocking=True),
        ]
        PREPROCESS_STREAM.synchronize()  # only queue codes that are ready
    
    # Get the global SNAC model instance
    audio_hat = await model_snac.batch_snac_model.acall({"codes": codes})
    audio_np = audio_hat.numpy(force=True)
    audio_bytes = (audio_np * 32767).astype(np.int16).tobytes()
    return audio_bytes


class OrpheusModel:
    """Orpheus TTS Model implementation for Modal integration"""
    
    def __init__(self, **kwargs) -> None:
        self._model = None
        self._tokenizer = None
        self.start_id = [128259]
        self.end_ids = [128009, 128260, 128261, 128257]

    def load(self, model_name="canopylabs/orpheus-3b-0.1-ft") -> None:
        """Load the Orpheus model and tokenizer with optimizations"""
        import os
        from transformers import AutoModelForCausalLM
        
        try:
            # Set up HuggingFace authentication
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if hf_token:
                logging.info("HuggingFace token found, setting up authentication")
                # Set token for huggingface_hub
                os.environ["HF_TOKEN"] = hf_token
                os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            else:
                logging.warning("No HuggingFace token found - this may cause issues with gated models")
            
            # Load tokenizer from cached location first, fallback to download
            try:
                tokenizer_path = "/root/.cache/huggingface/hub"
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    cache_dir=tokenizer_path,
                    local_files_only=False,
                    token=hf_token  # Pass token explicitly
                )
                logging.info(f"Loaded Orpheus tokenizer from {model_name}")
            except Exception as e:
                logging.warning(f"Failed to load tokenizer from cache, downloading: {e}")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    token=hf_token  # Pass token explicitly
                )
            
            # Load the actual Orpheus language model with optimizations
            try:
                logging.info(f"Loading Orpheus language model from {model_name} optimized for H100...")

                # Prefer BF16 on H100; avoid device_map=auto to prevent CPU offload
                preferred_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir="/root/.cache/huggingface/hub",
                    torch_dtype=preferred_dtype,
                    device_map=None,
                    token=hf_token,
                    trust_remote_code=True,
                    attn_implementation=self._get_best_attention_implementation(),
                    low_cpu_mem_usage=True,
                )
                if torch.cuda.is_available():
                    self._model = self._model.to("cuda")
                    logging.info("Orpheus model loaded fully on CUDA (%s)", preferred_dtype)
                else:
                    logging.info("Orpheus model loaded on CPU (%s)", preferred_dtype)

                self._model.eval()

                # Enable performance optimizations (TF32, attention backends, etc.)
                self._enable_performance_optimizations()

                # Memory/runtime optimizations for inference
                self._optimize_memory()

                # Optionally compile LLM only if explicitly enabled (reduce TTFT)
                self._compile_model()

                logging.info("Orpheus language model loaded and optimized successfully")
            except Exception as e:
                logging.error(f"Failed to load Orpheus language model: {e}")
                raise
            
            self.start_tokenized = (
                self._tokenizer.decode(self.start_id) + self._tokenizer.bos_token
            )
            self.end_tokenized = self._tokenizer.decode(self.end_ids)

            # Prefer numeric prompt building for correctness and speed
            self.use_fast_fmt = False
            logging.info("Orpheus tokenizer loaded successfully, using numeric prompt construction")

            # Initialize the global SNAC model
            global model_snac
            model_snac = SnacModelBatched()
            logging.info("SNAC model initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to load Orpheus model: {e}")
            raise

    def _get_best_attention_implementation(self):
        """Determine the best attention implementation available"""
        try:
            # Try Flash Attention 2 first (fastest) but handle CUDA symbol issues
            import flash_attn
            # Test if flash_attn actually works with current CUDA setup
            try:
                # Quick test to see if flash_attn can be used
                torch.cuda.is_available()  # Basic CUDA check
                logging.info("Flash Attention 2 available and compatible, using it")
                return "flash_attention_2"
            except Exception as e:
                logging.warning(f"Flash Attention 2 available but incompatible: {e}")
                raise ImportError("Flash attention compatibility issue")
        except (ImportError, Exception) as e:
            logging.info(f"Flash Attention 2 not usable: {e}")
        
        # Fallback to PyTorch's scaled_dot_product_attention (still fast)
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            logging.info("Using PyTorch scaled_dot_product_attention (SDPA)")
            return "sdpa"
        
        # Final fallback to default attention
        logging.info("Using default attention implementation")
        return None

    def _enable_performance_optimizations(self):
        """Enable various PyTorch performance optimizations"""
        try:
            # Enable TensorFloat32 for better H100 performance
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision('high')  # Use TF32 for float32 operations
                logging.info("TensorFloat32 (TF32) enabled for better H100 performance")
            
            # Enable optimized attention backends
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True) 
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            logging.info("Optimized attention backends enabled")
            
            # Set optimal CUDA settings
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.95)
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                logging.info("CUDA optimizations applied")
                
        except Exception as e:
            logging.warning(f"Performance optimization setup failed: {e}")

    def _compile_model(self):
        """Optionally compile the Orpheus model for inference.

        Disabled by default to minimize time-to-first-token; enable with ORPHEUS_COMPILE_LLM=1.
        """
        try:
            import os as _os
            if _os.environ.get("ORPHEUS_COMPILE_LLM", "0") != "1":
                logging.info("Skipping torch.compile for Orpheus LLM (set ORPHEUS_COMPILE_LLM=1 to enable)")
                return

            logging.info("Starting torch.compile optimization for Orpheus model...")
            compile_start = time.time()

            self._model = torch.compile(
                self._model,
                mode="reduce-overhead",
                dynamic=True,
                backend="inductor",
            )

            # Lightweight warmup to materialize graphs
            try:
                inputs = {
                    "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device="cuda" if torch.cuda.is_available() else None)
                }
                with torch.no_grad():
                    _ = self._model(**inputs)
                compile_time = time.time() - compile_start
                logging.info(f"torch.compile completed in {compile_time:.2f} seconds")
            except Exception as e:
                logging.warning(f"Model compilation warmup failed, continuing: {e}")
        except Exception as e:
            logging.warning(f"torch.compile failed, continuing without compilation: {e}")

    def _optimize_memory(self):
        """Apply memory/runtime optimizations to the model (inference-only)."""
        try:
            logging.info("Applying memory/runtime optimizations...")

            # Ensure KV cache is enabled for fast autoregressive decoding
            if hasattr(self._model.config, "use_cache"):
                self._model.config.use_cache = True
                logging.info("Model KV cache enabled")

            # Do NOT enable gradient checkpointing for inference
            if hasattr(self._model, "gradient_checkpointing_disable"):
                try:
                    self._model.gradient_checkpointing_disable()
                except Exception:
                    pass

            if torch.cuda.is_available():
                # Reserve most memory for the model while leaving headroom
                if hasattr(torch.cuda, "set_per_process_memory_fraction"):
                    torch.cuda.set_per_process_memory_fraction(0.96)
                torch.cuda.empty_cache()
        except Exception as e:
            logging.warning(f"Memory optimization failed, continuing: {e}")

    def _format_prompt_slow(self, prompt, voice="tara"):
        # Kept for compatibility; prefer numeric prompt ids via _build_prompt_ids
        if voice:
            adapted_prompt = f"{voice}: {prompt}"
        else:
            adapted_prompt = prompt
        input_ids = self._tokenizer.encode(adapted_prompt)
        full_ids = self.start_id + input_ids + self.end_ids
        return self._tokenizer.decode(full_ids)

    def _format_prompt_fast(self, prompt, voice="tara"):
        token_stream = self.start_tokenized
        if voice:
            token_stream += f"{voice}: "
        token_stream += prompt
        token_stream += self.end_tokenized
        return token_stream

    def format_prompt(self, prompt: str, voice="tara"):
        """Format the prompt for the model."""
        if self.use_fast_fmt:
            return self._format_prompt_fast(prompt, voice)
        else:
            return self._format_prompt_slow(prompt, voice)

    def _build_prompt_ids(self, prompt: str, voice: str = "tara") -> torch.Tensor:
        """Construct numeric prompt token ids for Orpheus (preferred)."""
        if voice:
            adapted_prompt = f"{voice}: {prompt}"
        else:
            adapted_prompt = prompt
        txt_ids: list[int] = self._tokenizer.encode(adapted_prompt)
        full_ids: list[int] = self.start_id + txt_ids + self.end_ids
        input_ids = torch.tensor([full_ids], dtype=torch.long)
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
        return input_ids

    async def generate_speech_stream(self, prompt: str, voice: str = "tara", **kwargs) -> AsyncIterator[bytes]:
        """Generate speech from text prompt and yield audio chunks"""
        try:
            req_id = str(kwargs.get("request_id", uuid.uuid4()))
            formatted_prompt = self.format_prompt(prompt, voice=voice)
            input_length = len(formatted_prompt)
            
            logging.info(f"Starting Orpheus TTS request_id {req_id} with input length {input_length}")
            
            if input_length > MAX_CHARACTERS_INPUT:
                raise ValueError(f"Prompt too long (len: {input_length}), max length is {MAX_CHARACTERS_INPUT} characters.")
            
            if self._model is None or self._tokenizer is None:
                raise RuntimeError("Orpheus model not loaded. Call load() first.")
            
            start_time = time.time()
            
            # Generate real audio tokens using the Orpheus model with optimized streaming
            async def generate_orpheus_tokens():
                """Generate speech tokens using optimized streaming with numeric prompts."""
                logging.info(
                    "Running Orpheus streaming inference for: '%s...'",
                    formatted_prompt[:100],
                )

                # Build numeric input ids to ensure correct special tokens and faster prefill
                inputs = {"input_ids": self._build_prompt_ids(prompt, voice)}

                with torch.no_grad():
                    max_new_tokens = int(kwargs.get("max_tokens", 2048))
                    temperature = float(kwargs.get("temperature", 0.7))
                    top_k = int(kwargs.get("top_k", 50))

                    gen_kwargs = dict(
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        do_sample=True,
                        pad_token_id=self._tokenizer.pad_token_id
                        or self._tokenizer.eos_token_id,
                        eos_token_id=self.end_ids,
                        use_cache=True,
                        num_beams=1,
                        early_stopping=True,
                        repetition_penalty=1.05,
                        no_repeat_ngram_size=3,
                    )

                    from transformers import TextIteratorStreamer
                    import threading

                    streamer = TextIteratorStreamer(
                        self._tokenizer,
                        timeout=60.0,
                        skip_prompt=True,
                        skip_special_tokens=False,
                    )

                    generation_thread = threading.Thread(
                        target=self._model.generate,
                        kwargs={**inputs, **gen_kwargs, "streamer": streamer},
                    )
                    generation_thread.start()

                    token_count = 0
                    for token_str in streamer:
                        if token_str is None:
                            break
                        token_count += 1
                        if "<custom_token_" in token_str:
                            yield token_str
                        if token_count % 64 == 0:
                            logging.debug("Generated %d tokens so far...", token_count)

                    generation_thread.join(timeout=60.0)
                    if generation_thread.is_alive():
                        logging.warning("Generation thread did not complete within timeout")
                    logging.info("Token generation finished (%d tokens)", token_count)
            
            token_gen = generate_orpheus_tokens()
            
            async for chunk in tokens_decoder(token_gen, req_id, start_time):
                yield chunk
                
        except Exception as e:
            logging.error(f"Error in Orpheus TTS request_id {req_id}: {e}")
            raise


# Global SNAC model instance
model_snac = None


# Initialize global model instance when module is imported
def initialize_orpheus_model():
    """Initialize the global Orpheus model instance"""
    global model_snac
    if model_snac is None:
        model_snac = SnacModelBatched()
    return model_snac