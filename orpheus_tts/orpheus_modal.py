"""
Orpheus TTS Modal Implementation

This module adapts the Orpheus TTS functionality for use with Modal,
providing a simplified interface for text-to-speech generation.
"""

import asyncio
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, List

import numpy as np
import torch
from fastapi.responses import StreamingResponse, Response
from transformers import AutoTokenizer

# Import batched and SNAC dependencies
import batched
from snac import SNAC

# Constants
_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")
SNAC_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SNAC_MAX_BATCH = 64
MAX_CHARACTERS_INPUT = 6144

# Force inference mode during the lifetime of the script
_inference_mode_raii_guard = torch._C._InferenceMode(True)


class SnacModelBatched:
    """Batched SNAC model for efficient audio generation."""
    
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
                import threading
                threading.Thread(target=self.compile, daemon=True).start()
            else:
                # Compile the model in the main thread
                self.compile()

    def compile(self):
        """Compile the model with torch.compile for better performance."""
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

    @batched.dynamically(batch_size=SNAC_MAX_BATCH, timeout_ms=15)
    def batch_snac_model(
        self, items: list[dict[str, list[torch.Tensor]]]
    ) -> list[torch.Tensor]:
        """Process batched SNAC model inference."""
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            all_codes = [codes["codes"] for codes in items]
            can_be_batched = len(items) > 1 and all(
                codes[0].shape == all_codes[0][0].shape for codes in all_codes
            )
            
            if can_be_batched:
                # stacked_codes = [(b,4), (b,8), (b,16)]
                stacked_codes = [
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
                    logging.warning(
                        "Warning: items can't be batched, using individual decoding."
                    )
                
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


class OrpheusTTSModel:
    """Main Orpheus TTS model class for Modal."""
    
    def __init__(self):
        self._tokenizer = None
        self.snac_model = None
        self.start_id = [128259]
        self.end_ids = [128009, 128260, 128261, 128257]
        self.preprocess_stream = torch.Stream(SNAC_DEVICE)
        
    def load(self):
        """Load the model and tokenizer."""
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained("baseten/orpheus-3b-0.1-ft")
        
        # Set up tokenized strings for fast formatting
        self.start_tokenized = (
            self._tokenizer.decode(self.start_id) + self._tokenizer.bos_token
        )
        self.end_tokenized = self._tokenizer.decode(self.end_ids)
        
        # Initialize SNAC model
        self.snac_model = SnacModelBatched()
        
        # Test fast formatting
        self.use_fast_fmt = self._format_prompt_fast(
            "hello world", "tara"
        ) == self._format_prompt_slow("hello world", "tara")
        
        logging.info("Orpheus TTS model loaded successfully")

    def _format_prompt_slow(self, prompt, voice="tara"):
        """Slow prompt formatting using tokenizer encode/decode."""
        if voice:
            adapted_prompt = f"{voice}: {prompt}"
        else:
            adapted_prompt = prompt
        input_ids = self._tokenizer.encode(adapted_prompt)
        full_ids = self.start_id + input_ids + self.end_ids
        return self._tokenizer.decode(full_ids)

    def _format_prompt_fast(self, prompt, voice="tara"):
        """Fast prompt formatting using string concatenation."""
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

    def turn_token_into_id(self, token_string: int, index: int):
        """Extract and convert the last custom token ID from a string."""
        return token_string - 10 - ((index % 7) * 4096)

    def split_custom_tokens(self, s: str) -> List[int]:
        """Extracts all substrings enclosed in <custom_token_…> from the input string."""
        matches = _TOKEN_RE.findall(s)
        return [int(match) for match in matches if match != "0"]

    @torch.inference_mode()
    async def convert_to_audio(self, frame_ids: list[int]) -> bytes | None:
        """Convert a list of token IDs into audio bytes efficiently."""
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
            
        with torch.cuda.stream(self.preprocess_stream):
            codes = [
                codes_0.unsqueeze(0).to(SNAC_DEVICE),
                codes_1.unsqueeze(0).to(SNAC_DEVICE),
                codes_2.unsqueeze(0).to(SNAC_DEVICE),
            ]
            self.preprocess_stream.synchronize()  # only queue codes that are ready
            
        audio_hat = await self.snac_model.batch_snac_model.acall({"codes": codes})
        audio_np = audio_hat.numpy(force=True)
        audio_bytes = (audio_np * 32767).astype(np.int16).tobytes()
        return audio_bytes

    async def tokens_decoder(
        self, token_gen: Iterator, request_id: str, start_time: int
    ) -> Iterator[bytes]:
        """Decoder that pipelines convert_to_audio calls but enforces strict in-order yields."""
        assert hasattr(token_gen, "__aiter__")
        audio_queue = asyncio.Queue()

        async def producer(token_gen: Iterator):
            buffer: list[int] = []
            count = 0
            tft = 0
            async for token_sim in token_gen:
                if tft == 0:
                    tft = time.time()
                for tok_str in self.split_custom_tokens(token_sim):
                    token = self.turn_token_into_id(int(tok_str), count)
                    buffer.append(token)
                    count += 1
                    # every 7 tokens → one frame; once we have at least 28 tokens, we extract the last 28
                    if count % 7 == 0 and count > 27:
                        buf_to_proc = buffer[-28:]
                        task = asyncio.create_task(self.convert_to_audio(buf_to_proc))
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
            task = await audio_queue.get()
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

    async def generate_speech(self, text: str, voice: str = "tara") -> bytes:
        """Generate speech from text using Hugging Face Transformers."""
        if len(text) > MAX_CHARACTERS_INPUT:
            raise ValueError(f"Text too long: {len(text)} > {MAX_CHARACTERS_INPUT}")
            
        # Format the prompt
        formatted_prompt = self.format_prompt(text, voice)
        
        try:
            # Use the Hugging Face model for token generation
            if not hasattr(self, '_hf_model'):
                logging.info("Loading Hugging Face model for inference...")
                from transformers import AutoModelForCausalLM
                
                self._hf_model = AutoModelForCausalLM.from_pretrained(
                    "baseten/orpheus-3b-0.1-ft",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                logging.info("Hugging Face model loaded")
            
            # Tokenize input
            input_ids = self._tokenizer.encode(formatted_prompt, return_tensors="pt")
            input_ids = input_ids.to(self._hf_model.device)
            
            # Generate tokens
            with torch.no_grad():
                outputs = self._hf_model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 4096,  # max_tokens
                    temperature=0.6,
                    top_p=0.8,
                    do_sample=True,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=128258,  # end_id from original
                    repetition_penalty=1.1,
                )
            
            # Extract generated tokens (remove input)
            generated_tokens = outputs[0][input_ids.shape[1]:]
            generated_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=False)
            
            # Process tokens to audio using the existing pipeline
            audio_chunks = []
            buffer = []
            count = 0
            
            # Extract custom tokens and convert to audio
            for tok_str in self.split_custom_tokens(generated_text):
                token = self.turn_token_into_id(int(tok_str), count)
                buffer.append(token)
                count += 1
                
                # Process every 28 tokens (4 frames)
                if count % 7 == 0 and count > 27:
                    buf_to_proc = buffer[-28:]
                    audio_bytes = await self.convert_to_audio(buf_to_proc)
                    if audio_bytes:
                        audio_chunks.append(audio_bytes)
            
            # Combine all audio chunks
            if audio_chunks:
                return b''.join(audio_chunks)
            else:
                logging.warning("No audio generated from tokens")
                return b""
                
        except Exception as e:
            logging.error(f"Error in generate_speech: {e}")
            # Return a simple tone as fallback
            return self._generate_fallback_audio(text)
    
    def _generate_fallback_audio(self, text: str) -> bytes:
        """Generate a simple fallback audio (sine wave) when model fails."""
        import math
        
        # Generate a simple sine wave based on text length
        duration = min(len(text) * 0.1, 10.0)  # 0.1s per character, max 10s
        sample_rate = 24000
        frequency = 440.0  # A4 note
        
        samples = []
        for i in range(int(duration * sample_rate)):
            t = i / sample_rate
            sample = int(16383 * math.sin(2 * math.pi * frequency * t))
            samples.append(sample)
        
        # Convert to bytes
        import struct
        audio_bytes = b''.join(struct.pack('<h', sample) for sample in samples)
        
        logging.info(f"Generated fallback audio: {len(audio_bytes)} bytes for {duration:.1f}s")
        return audio_bytes


# Global model instance
orpheus_model = None


def get_model():
    """Get or initialize the global model instance."""
    global orpheus_model
    if orpheus_model is None:
        orpheus_model = OrpheusTTSModel()
        orpheus_model.load()
    return orpheus_model