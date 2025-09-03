"""
Orpheus TTS Model Implementation
Adapted from: https://github.com/basetenlabs/truss-examples/tree/main/orpheus-best-performance

This module provides an optimized Orpheus TTS implementation with batching and quantization
for high-performance text-to-speech generation.
"""

from typing import Any, Iterator, List, Awaitable
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

    @batched.dynamically(batch_size=SNAC_MAX_BATCH, timeout_ms=15)
    def batch_snac_model(
        self, items: list[dict[str, list[torch.Tensor]]]
    ) -> list[torch.Tensor]:
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
    token_gen: Iterator, request_id: str, start_time: int
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


# Global instance for batched processing
model_snac = None


def get_snac_model():
    """Get or create the global SNAC model instance."""
    global model_snac
    if model_snac is None:
        model_snac = SnacModelBatched()
    return model_snac


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
    
    snac_model = get_snac_model()
    audio_hat = await snac_model.batch_snac_model.acall({"codes": codes})
    audio_np = audio_hat.numpy(force=True)
    audio_bytes = (audio_np * 32767).astype(np.int16).tobytes()
    return audio_bytes


class OrpheusModel:
    """Orpheus TTS Model implementation for Modal deployment."""
    
    def __init__(self, **kwargs) -> None:
        self._secrets = kwargs.get("secrets", {})
        self._data_dir = kwargs.get("data_dir", "/app")
        self._model = None
        self._tokenizer = None
        self.start_id = [128259]
        self.end_ids = [128009, 128260, 128261, 128257]
        
        # Initialize SNAC model for audio generation
        self.snac_model = None

    def load(self) -> None:
        """Load the Orpheus model and tokenizer."""
        logger.info("Loading Orpheus model...")
        
        # Load tokenizer
        try:
            self._tokenizer = AutoTokenizer.from_pretrained("baseten/orpheus-3b-0.1-ft")
        except Exception as e:
            logger.warning(f"Failed to load from baseten/orpheus-3b-0.1-ft, trying canopylabs: {e}")
            self._tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-ft")
            
        self.start_tokenized = (
            self._tokenizer.decode(self.start_id) + self._tokenizer.bos_token
        )
        self.end_tokenized = self._tokenizer.decode(self.end_ids)

        self.use_fast_fmt = self._format_prompt_fast(
            "hello world", "tara"
        ) == self._format_prompt_slow("hello world", "tara")
        
        # Initialize SNAC model
        self.snac_model = get_snac_model()
        
        logger.info("Orpheus model loaded successfully")

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

    async def generate_speech(self, text: str, voice: str = "tara") -> AsyncIterator[bytes]:
        """Generate speech audio from text using Orpheus model."""
        try:
            req_id = str(uuid.uuid4())
            formatted_prompt = self.format_prompt(text, voice=voice)
            input_length = len(formatted_prompt)
            
            logger.info(f"Starting Orpheus TTS request {req_id} with input length {input_length}")
            
            if input_length > MAX_CHARACTERS_INPUT:
                raise ValueError(f"Input too long: {input_length} > {MAX_CHARACTERS_INPUT}")
            
            # Model parameters
            model_input = {
                "prompt": formatted_prompt,
                "temperature": 0.6,
                "top_p": 0.8,
                "max_tokens": 6144,
                "end_id": 128258,
                "repetition_penalty": 1.1,
                "voice": voice,
                "request_id": req_id
            }
            
            start_time = time.time()
            
            # For now, we'll use a placeholder token generation
            # In a real implementation, this would connect to the TensorRT-LLM engine
            async def mock_token_generator():
                """Mock token generator - replace with actual Orpheus LLM inference."""
                # This is a placeholder - in real implementation, this would be
                # connected to the TensorRT-LLM engine from the config.yaml
                mock_tokens = [
                    "<custom_token_128270>", "<custom_token_128271>", "<custom_token_128272>",
                    "<custom_token_128273>", "<custom_token_128274>", "<custom_token_128275>",
                    "<custom_token_128276>", "<custom_token_128277>", "<custom_token_128278>",
                    "<custom_token_128279>", "<custom_token_128280>", "<custom_token_128281>",
                    "<custom_token_128282>", "<custom_token_128283>", "<custom_token_128284>",
                    "<custom_token_128285>", "<custom_token_128286>", "<custom_token_128287>",
                    "<custom_token_128288>", "<custom_token_128289>", "<custom_token_128290>",
                    "<custom_token_128291>", "<custom_token_128292>", "<custom_token_128293>",
                    "<custom_token_128294>", "<custom_token_128295>", "<custom_token_128296>",
                    "<custom_token_128297>", "<custom_token_128298>", "<custom_token_128299>",
                ]
                for token in mock_tokens:
                    yield token
                    await asyncio.sleep(0.05)  # Simulate generation delay
            
            # Use the tokens decoder to convert tokens to audio
            async for audio_chunk in tokens_decoder(mock_token_generator(), req_id, start_time):
                yield audio_chunk
                
        except Exception as e:
            logger.error(f"Error in Orpheus TTS generation for request {req_id}: {e}")
            raise


class OrpheusModelWithEngine(OrpheusModel):
    """Orpheus model with TensorRT-LLM engine integration."""
    
    def __init__(self, trt_llm=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._engine = trt_llm["engine"] if trt_llm else None

    async def predict(
        self, model_input: Any, request: fastapi.Request
    ) -> StreamingResponse:
        """Main prediction endpoint for Orpheus TTS."""
        try:
            req_id = str(model_input.get("request_id", uuid.uuid4()))
            model_input["prompt"] = self.format_prompt(
                model_input["prompt"], voice=model_input.get("voice", "tara")
            )
            input_length = len(model_input["prompt"])
            logging.info(
                f"Starting request_id {req_id} with input length {input_length}"
            )
            if input_length > MAX_CHARACTERS_INPUT:
                return Response(
                    (
                        f"Your suggested prompt is too long (len: {input_length}), max length is {MAX_CHARACTERS_INPUT} characters."
                        "To generate audio faster, please split your request into multiple prompts. "
                    ),
                    status_code=400,
                )
            model_input["temperature"] = model_input.get("temperature", 0.6)
            model_input["top_p"] = model_input.get("top_p", 0.8)
            model_input["max_tokens"] = model_input.get("max_tokens", 6144)
            if model_input.get("end_id") is not None:
                logging.info(
                    "Not using end_id from model_input:", model_input["end_id"]
                )
            model_input["end_id"] = 128258
            model_input["repetition_penalty"] = model_input.get(
                "repetition_penalty", 1.1
            )
            start_time = time.time()

            async def audio_stream(req_id: str):
                if self._engine:
                    token_gen = await self._engine.predict(model_input, request)
                    if isinstance(token_gen, StreamingResponse):
                        token_gen = token_gen.body_iterator
                else:
                    # Fallback to mock generation if no engine available
                    async def mock_generator():
                        mock_tokens = [f"<custom_token_{128270 + i}>" for i in range(28)]
                        for token in mock_tokens:
                            yield token
                            await asyncio.sleep(0.05)
                    token_gen = mock_generator()

                async for chunk in tokens_decoder(token_gen, req_id, start_time):
                    yield chunk

            return StreamingResponse(
                audio_stream(req_id),
                media_type="audio/wav",
                headers={"X-Baseten-Input-Tokens": str(input_length)},
            )
        except Exception as e:
            print(f"Error in request_id {req_id}: {e} with input {model_input}")
            return Response(
                f"An internal server error occurred while processing your request {req_id}",
                status_code=500,
            )