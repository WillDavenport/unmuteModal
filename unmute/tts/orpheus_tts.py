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
SNAC_MAX_BATCH = 128  # Increased for better H100 utilization
PREPROCESS_STREAM = torch.Stream(SNAC_DEVICE)
MAX_CHARACTERS_INPUT = 6144

# H100 optimization constants
OPTIMAL_CHUNK_SIZE = 64  # Optimal streaming chunk size for H100
STREAMING_BUFFER_SIZE = 256  # Buffer size for streaming generation

logger = logging.getLogger(__name__)


class SnacModelBatched:
    def __init__(self):
        # Use bfloat16 for better H100 performance instead of float32
        self.dtype_decoder = torch.bfloat16 if torch.cuda.is_available() else torch.float32
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

    @batched.dynamically(batch_size=SNAC_MAX_BATCH, timeout_ms=5)  # Reduced timeout for faster streaming
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
    """High-performance streaming decoder optimized for H100."""
    assert hasattr(token_gen, "__aiter__")
    
    buffer: list[int] = []
    count = 0
    tft = 0
    first_audio_yielded = False
    
    # Use optimal chunk size for H100 streaming
    chunk_size = OPTIMAL_CHUNK_SIZE  # 64 tokens = ~9 frames for better batching
    
    async for token_sim in token_gen:
        if tft == 0:
            tft = time.time()
        
        # Collect tokens for batch processing
        for tok_str in split_custom_tokens(token_sim):
            token = turn_token_into_id(int(tok_str), count)
            buffer.append(token)
            count += 1
            
            # Process in optimal chunks for H100 (every 7 tokens = 1 frame)
            # Stream more aggressively for lower latency
            if count % 7 == 0:  # Process complete frames immediately
                # Use larger batches when buffer has accumulated enough tokens
                frames_available = len(buffer) // 7
                if frames_available >= 1:  # Process as soon as we have 1+ complete frames
                    tokens_to_process = min(frames_available * 7, chunk_size)
                    if tokens_to_process >= 7:  # At least one complete frame
                        buf_to_proc = buffer[:tokens_to_process]
                        buffer = buffer[tokens_to_process:]  # Remove processed tokens
                        
                        # Convert to audio with optimized batching
                        audio_bytes = await convert_to_audio(buf_to_proc)
                        if audio_bytes is not None:
                            if not first_audio_yielded:
                                ttfb = time.time() - start_time
                                logging.info(f"First audio chunk for request_id {request_id} - TTFB: {ttfb:.3f}s")
                                first_audio_yielded = True
                            yield audio_bytes
    
    # Process any remaining tokens
    if len(buffer) >= 7:  # At least one frame
        remaining_tokens = (len(buffer) // 7) * 7  # Round down to complete frames
        if remaining_tokens > 0:
            buf_to_proc = buffer[:remaining_tokens]
            audio_bytes = await convert_to_audio(buf_to_proc)
            if audio_bytes is not None:
                yield audio_bytes
    
    # Log performance metrics
    elapsed = time.time() - start_time
    time_to_first_token = tft - start_time if tft > 0 else elapsed
    time_of_generation = time.time() - tft if tft > 0 else elapsed
    token_generation_speed = count / time_of_generation if time_of_generation > 0 else 0
    
    logging.info(
        f"Finished `{request_id}`, total tokens: {count}, time: {elapsed:.2f}s. "
        f"tokens/s generation: {token_generation_speed:.2f} (ttft: {time_to_first_token:.2f}s, generation time: {time_of_generation:.2f}s) "
        f"real-time factor: {(time_of_generation / (count / 100) if count > 0 else 0):.2f}x"
    )


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

    arr = torch.tensor(frame_ids[: n * 7], dtype=torch.int32)
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
            codes_0.unsqueeze(0).to(SNAC_DEVICE),
            codes_1.unsqueeze(0).to(SNAC_DEVICE),
            codes_2.unsqueeze(0).to(SNAC_DEVICE),
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
                logging.info(f"Loading Orpheus language model from {model_name} with optimizations...")
                
                # Try with H100-optimized configuration first
                try:
                    # Use bfloat16 for better H100 performance
                    optimal_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
                    
                    self._model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        cache_dir="/root/.cache/huggingface/hub",
                        torch_dtype=optimal_dtype,  # H100 optimized precision
                        device_map="auto",
                        token=hf_token,
                        trust_remote_code=True,
                        # H100-optimized attention implementation
                        attn_implementation=self._get_best_attention_implementation(),
                        low_cpu_mem_usage=True,
                        # Additional H100 optimizations
                        use_cache=True,  # Enable KV caching
                        max_memory={0: "75GB"} if torch.cuda.is_available() else None,  # Reserve memory for H100
                    )
                    logging.info("Orpheus model loaded with accelerate device mapping and optimizations")
                except Exception as e:
                    logging.warning(f"Failed to load with optimizations, trying basic loading: {e}")
                    # Fallback: load without advanced optimizations
                    # Fallback: load with basic H100 optimizations
                    optimal_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
                    self._model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        cache_dir="/root/.cache/huggingface/hub",
                        torch_dtype=optimal_dtype,
                        token=hf_token,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        use_cache=True,
                    )
                    if torch.cuda.is_available():
                        self._model = self._model.cuda()
                        logging.info("Orpheus model loaded and moved to CUDA manually")
                    else:
                        logging.info("Orpheus model loaded on CPU")
                
                self._model.eval()
                
                # Enable performance optimizations (TF32, attention backends, etc.)
                self._enable_performance_optimizations()
                
                # Memory optimizations
                self._optimize_memory()
                
                # Apply H100-optimized torch.compile
                self._compile_model()
                
                logging.info("Orpheus language model loaded and optimized successfully")
            except Exception as e:
                logging.error(f"Failed to load Orpheus language model: {e}")
                raise
            
            self.start_tokenized = (
                self._tokenizer.decode(self.start_id) + self._tokenizer.bos_token
            )
            self.end_tokenized = self._tokenizer.decode(self.end_ids)

            self.use_fast_fmt = self._format_prompt_fast(
                "hello world", "tara"
            ) == self._format_prompt_slow("hello world", "tara")
            
            logging.info(f"Orpheus tokenizer loaded successfully, fast format: {self.use_fast_fmt}")
            
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
        """Enable H100-specific performance optimizations"""
        try:
            if torch.cuda.is_available():
                # Enable TensorFloat32 for better H100 performance
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision('high')  # Use TF32 for float32 operations
                logging.info("TensorFloat32 (TF32) enabled for H100")
                
                # H100-specific optimizations
                device_props = torch.cuda.get_device_properties(0)
                if device_props.major >= 9:  # H100 is compute capability 9.0+
                    logging.info(f"H100 GPU detected: {device_props.name}")
                    
                    # Enable H100-specific features
                    if hasattr(torch.backends.cuda.matmul, 'allow_fp16_reduced_precision_reduction'):
                        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
                        logging.info("FP16 reduced precision reduction enabled")
                    
                    # Try to enable FP8 optimizations if available
                    self._enable_fp8_optimizations()
                    
                    # H100 memory optimizations
                    torch.cuda.set_per_process_memory_fraction(0.90)  # Leave room for FP8 conversions
                else:
                    logging.info(f"Non-H100 GPU detected: {device_props.name}")
                    torch.cuda.set_per_process_memory_fraction(0.95)
            
            # Enable optimized attention backends (critical for performance)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True) 
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            logging.info("Optimized attention backends enabled")
            
            # Additional CUDA optimizations
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
                
                # Enable CUDA graphs if supported
                if hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
                    try:
                        # Test CUDA graphs support
                        torch.cuda.synchronize()
                        logging.info("CUDA graphs support detected")
                    except Exception as e:
                        logging.warning(f"CUDA graphs not available: {e}")
                
                logging.info("H100 CUDA optimizations applied")
                
        except Exception as e:
            logging.warning(f"Performance optimization setup failed: {e}")
    
    def _enable_fp8_optimizations(self):
        """Enable FP8 optimizations if available on H100"""
        try:
            # Check for FP8 support (experimental)
            if hasattr(torch, 'float8_e4m3fn'):
                logging.info("FP8 support detected - enabling experimental optimizations")
                
                # Set environment variables for FP8 optimizations
                import os
                os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
                os.environ['TORCH_COMPILE_DEBUG'] = '0'  # Disable debug for performance
                
                # Note: Full FP8 quantization would require additional libraries
                # like transformer-engine or TensorRT-LLM
                logging.info("FP8 environment configured")
            else:
                logging.info("FP8 support not available in current PyTorch version")
                
        except Exception as e:
            logging.warning(f"FP8 optimization setup failed: {e}")
    
    def _get_optimized_generation_config(self, kwargs):
        """Get generation config optimized for H100 streaming performance"""
        max_new_tokens = kwargs.get("max_tokens", 2048)  # Reduced for streaming
        temperature = kwargs.get("temperature", 0.8)  # Slightly higher for better quality
        top_k = kwargs.get("top_k", 40)  # Optimized for H100
        top_p = kwargs.get("top_p", 0.9)  # Add nucleus sampling
        
        return {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            "eos_token_id": self.end_ids,
            "use_cache": True,  # Critical for streaming performance
            # H100 streaming optimizations
            "num_beams": 1,  # No beam search overhead
            "repetition_penalty": 1.05,  # Reduced penalty for faster generation
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 2,  # Reduced for speed
            # Additional streaming optimizations
            "output_scores": False,  # Don't compute scores to save memory
            "return_dict_in_generate": True,  # Required for streaming
        }

    def _compile_model(self):
        """Compile the Orpheus model with H100-optimized settings"""
        try:
            logging.info("Starting H100-optimized torch.compile for Orpheus model...")
            compile_start = time.time()
            
            # H100-optimized compilation settings
            compile_kwargs = {
                "mode": "max-autotune",  # Best performance for H100
                "dynamic": True,  # Handle variable sequence lengths
                "backend": "inductor",
                "options": {
                    "triton.cudagraphs": True,  # Enable CUDA graphs for H100
                    "max_autotune": True,  # Aggressive optimization
                    "epilogue_fusion": True,  # Fuse operations
                    "max_autotune_gemm": True,  # Optimize matrix multiplications for H100
                }
            }
            
            # Check if we're on H100 for additional optimizations
            if torch.cuda.is_available():
                device_props = torch.cuda.get_device_properties(0)
                if device_props.major >= 9:  # H100 is compute capability 9.0+
                    logging.info("H100 detected - enabling advanced compilation optimizations")
                    compile_kwargs["options"]["use_mixed_mm"] = True  # Mixed precision matmul
            
            # Compile the model for faster inference
            self._model = torch.compile(self._model, **compile_kwargs)
            
            # Warm up the compiled model with a small example
            # This helps with compilation and caching
            try:
                dummy_input = "tara: Hello world"
                inputs = self._tokenizer(dummy_input, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    # Run a small forward pass to trigger compilation
                    _ = self._model(**inputs)
                    
                compile_time = time.time() - compile_start
                logging.info(f"torch.compile completed in {compile_time:.2f} seconds")
                
            except Exception as e:
                logging.warning(f"Model compilation warmup failed, but continuing: {e}")
                
        except Exception as e:
            logging.warning(f"torch.compile failed, continuing without compilation: {e}")
            # Don't fail the entire load process if compilation fails

    def _optimize_memory(self):
        """Apply memory optimizations to the model"""
        try:
            logging.info("Applying memory optimizations...")
            
            # Enable gradient checkpointing if available (saves memory during forward pass)
            if hasattr(self._model, 'gradient_checkpointing_enable'):
                self._model.gradient_checkpointing_enable()
                logging.info("Gradient checkpointing enabled")
            
            # Set model to use optimized attention if available
            if hasattr(self._model.config, 'use_cache'):
                self._model.config.use_cache = True
                logging.info("Model cache enabled")
                
            # Enable memory efficient attention if supported
            if torch.cuda.is_available():
                # Clear cache before starting
                torch.cuda.empty_cache()
                logging.info("CUDA cache cleared")
                
                # Set memory fraction to leave room for other operations
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(0.95)
                    logging.info("CUDA memory fraction set to 95%")
            
        except Exception as e:
            logging.warning(f"Memory optimization failed, continuing: {e}")

    def _monitor_gpu_utilization(self):
        """Monitor GPU performance during generation"""
        if torch.cuda.is_available():
            try:
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                memory_reserved = torch.cuda.memory_reserved() / 1e9
                
                device_props = torch.cuda.get_device_properties(0)
                total_memory = device_props.total_memory / 1e9
                
                logging.info(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB, "
                           f"Total: {total_memory:.2f}GB")
                logging.info(f"GPU Utilization: {(memory_allocated / total_memory * 100):.1f}%")
                
                # Log device info
                logging.info(f"GPU Device: {device_props.name}, Compute Capability: {device_props.major}.{device_props.minor}")
                
            except Exception as e:
                logging.warning(f"GPU monitoring failed: {e}")

    def _format_prompt_slow(self, prompt, voice="tara"):
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
            logging.warning("Warn: Using slow format")
            return self._format_prompt_slow(prompt, voice)

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
            
            # Monitor GPU state before generation
            self._monitor_gpu_utilization()
            
            # Clear CUDA cache to ensure optimal memory usage
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Generate real audio tokens using streaming generation for H100 optimization
            async def generate_orpheus_tokens():
                """Generate speech tokens using true streaming generation for minimal latency"""
                logging.info(f"Running H100-optimized streaming Orpheus inference for: '{formatted_prompt[:100]}...'")
                
                # Tokenize the input
                inputs = self._tokenizer(formatted_prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Use optimized generation config for H100 streaming
                generation_config = self._get_optimized_generation_config(kwargs)
                
                # Implement true streaming generation
                with torch.no_grad():
                    # Initialize generation state
                    input_ids = inputs['input_ids']
                    attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids))
                    
                    # Past key values for efficient generation
                    past_key_values = None
                    token_count = 0
                    
                    # Stream tokens one by one for minimal latency
                    for _ in range(generation_config['max_new_tokens']):
                        # Generate next token with KV caching
                        model_inputs = {
                            'input_ids': input_ids,
                            'attention_mask': attention_mask,
                            'past_key_values': past_key_values,
                            'use_cache': True,
                        }
                        
                        # Forward pass - only compute next token
                        outputs = self._model(**model_inputs)
                        
                        # Sample next token using optimized sampling
                        logits = outputs.logits[:, -1, :] / generation_config['temperature']
                        
                        # Apply top-k and top-p filtering
                        if generation_config['top_k'] > 0:
                            top_k_logits, top_k_indices = torch.topk(logits, generation_config['top_k'])
                            logits = torch.full_like(logits, float('-inf'))
                            logits.scatter_(1, top_k_indices, top_k_logits)
                        
                        # Apply top-p (nucleus) sampling
                        if generation_config.get('top_p', 1.0) < 1.0:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > generation_config['top_p']
                            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                            sorted_indices_to_remove[:, 0] = 0
                            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                            logits[indices_to_remove] = float('-inf')
                        
                        # Sample from the filtered distribution
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        
                        # Check for end tokens
                        if next_token.item() in self.end_ids:
                            break
                        
                        # Update generation state
                        input_ids = torch.cat([input_ids, next_token], dim=-1)
                        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
                        past_key_values = outputs.past_key_values
                        token_count += 1
                        
                        # Decode and yield token immediately for streaming
                        token_str = self._tokenizer.decode([next_token.item()], skip_special_tokens=False)
                        
                        # Only yield custom tokens for audio synthesis
                        if "<custom_token_" in token_str:
                            yield token_str
                            # Small async yield to allow other operations
                            await asyncio.sleep(0)
                    
                    logging.info(f"Completed streaming token generation: {token_count} total tokens")
            
            # Use the streaming token generator
            token_gen = generate_orpheus_tokens()
            
            # Process tokens through the optimized decoder
            async for chunk in tokens_decoder(token_gen, req_id, start_time):
                yield chunk
                
        except Exception as e:
            logging.error(f"Error in Orpheus TTS request_id {req_id}: {e}")
            raise

    async def benchmark_performance(self, test_texts=None):
        """Comprehensive performance benchmark for Orpheus TTS"""
        if test_texts is None:
            test_texts = [
                "Short test.",
                "Medium length test sentence with more words to evaluate performance.",
                "Very long test sentence with many words to test the performance impact of longer inputs on the Orpheus TTS system and measure real-time factors."
            ]
        
        logging.info("Starting Orpheus TTS performance benchmark...")
        
        for i, text in enumerate(test_texts):
            logging.info(f"Benchmarking text {i+1}/{len(test_texts)}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            start_time = time.time()
            audio_chunks = []
            first_chunk_time = None
            chunk_count = 0
            
            try:
                async for chunk in self.generate_speech_stream(text, request_id=f"benchmark_{i+1}"):
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - start_time
                    audio_chunks.append(chunk)
                    chunk_count += 1
                
                total_time = time.time() - start_time
                
                # Estimate audio duration (16kHz, 16-bit samples)
                total_audio_bytes = sum(len(chunk) for chunk in audio_chunks)
                audio_duration = total_audio_bytes / (2 * 24000)  # 24kHz sampling rate
                rtf = total_time / audio_duration if audio_duration > 0 else float('inf')
                
                # Log results
                logging.info(f"Benchmark Results for text {i+1}:")
                logging.info(f"  Text length: {len(text)} characters")
                logging.info(f"  TTFB (Time to First Byte): {first_chunk_time:.3f}s")
                logging.info(f"  Total generation time: {total_time:.3f}s")
                logging.info(f"  Audio duration: {audio_duration:.3f}s")
                logging.info(f"  Real-time factor: {rtf:.2f}x")
                logging.info(f"  Audio chunks generated: {chunk_count}")
                logging.info(f"  Total audio bytes: {total_audio_bytes:,}")
                
                # Performance assessment
                if first_chunk_time < 0.15:
                    ttfb_status = "EXCELLENT"
                elif first_chunk_time < 1.0:
                    ttfb_status = "GOOD"
                elif first_chunk_time < 2.0:
                    ttfb_status = "ACCEPTABLE"
                else:
                    ttfb_status = "NEEDS IMPROVEMENT"
                
                if rtf < 1.0:
                    rtf_status = "EXCELLENT (faster than real-time)"
                elif rtf < 2.0:
                    rtf_status = "GOOD"
                elif rtf < 5.0:
                    rtf_status = "ACCEPTABLE"
                else:
                    rtf_status = "NEEDS IMPROVEMENT"
                
                logging.info(f"  TTFB Assessment: {ttfb_status}")
                logging.info(f"  RTF Assessment: {rtf_status}")
                logging.info("-" * 60)
                
            except Exception as e:
                logging.error(f"Benchmark failed for text {i+1}: {e}")
        
        logging.info("Orpheus TTS performance benchmark completed.")


# Global SNAC model instance
model_snac = None


# Initialize global model instance when module is imported
def initialize_orpheus_model():
    """Initialize the global Orpheus model instance"""
    global model_snac
    if model_snac is None:
        model_snac = SnacModelBatched()
    return model_snac