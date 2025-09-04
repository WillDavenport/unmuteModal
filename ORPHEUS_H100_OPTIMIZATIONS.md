# 🚀 Orpheus TTS H100 Performance Optimizations

## 📊 Performance Issue Analysis

**Original Performance:**
- Real-time factor: **15.68x** (extremely slow)
- Time to first audio chunk: **8.28s** (should be <150ms)
- Total generation time: **69.60s** for ~4.4s of audio
- Expected H100 performance: **<1.0x RTF** (faster than real-time)

## 🔧 Key Optimizations Implemented

### 1. **Streaming Generation Architecture**
**Before:** Batch generation using `model.generate()` - processes entire sequence then iterates
**After:** True streaming generation - yields tokens as they're generated

```python
# NEW: Streaming token generation with KV caching
for _ in range(max_new_tokens):
    outputs = self._model(**model_inputs)
    # Sample and yield immediately
    next_token = sample_token(outputs.logits)
    yield next_token
    # Update KV cache for next iteration
    past_key_values = outputs.past_key_values
```

### 2. **H100-Optimized Model Configuration**
- **Precision:** `torch.bfloat16` instead of `torch.float16` (H100 optimized)
- **SNAC Decoder:** `torch.bfloat16` instead of `torch.float32` (50% memory reduction)
- **Attention:** Flash Attention 2 with SDPA fallbacks
- **Memory:** Optimized allocation for H100's 80GB HBM3

### 3. **Advanced Torch.Compile Settings**
```python
compile_kwargs = {
    "mode": "max-autotune",  # Best performance for H100
    "dynamic": True,
    "backend": "inductor",
    "options": {
        "triton.cudagraphs": True,  # Enable CUDA graphs
        "max_autotune_gemm": True,  # Optimize matmul for H100
        "epilogue_fusion": True,    # Fuse operations
    }
}
```

### 4. **Enhanced Batching and Streaming**
- **SNAC Batch Size:** Increased from 64 to 128
- **Streaming Chunks:** Optimized to 64 tokens (9 frames)
- **Timeout:** Reduced from 15ms to 5ms for faster streaming
- **Buffer Management:** Immediate processing of complete frames

### 5. **H100-Specific Performance Features**
- **TensorFloat32 (TF32):** Enabled for 19x speedup on H100
- **FP8 Support:** Experimental FP8 optimizations where available
- **CUDA Graphs:** Enabled for reduced kernel launch overhead
- **Memory Management:** Optimized for H100's memory bandwidth

### 6. **Modal Deployment Optimizations**
- **CPU:** Increased from 4.0 to 8.0 cores for async processing
- **Memory:** Increased from 32GB to 40GB for optimization overhead
- **Concurrency:** Increased from 10 to 16 max inputs
- **Keep-Warm:** 1 container always ready for instant response
- **Environment Variables:** H100-specific CUDA optimizations

## 🎯 Expected Performance Improvements

### Time to First Byte (TTFB)
- **Before:** 8.28s
- **Target:** <150ms on H100
- **Improvement:** ~55x faster

### Real-Time Factor (RTF)
- **Before:** 15.68x (very slow)
- **Target:** <1.0x (faster than real-time)
- **Improvement:** >15x faster

### Memory Efficiency
- **SNAC Model:** 50% memory reduction (bfloat16 vs float32)
- **Model Loading:** Optimized device mapping and memory allocation
- **Streaming:** Reduced memory footprint through immediate processing

## 🧪 Testing & Validation

### Run Performance Tests
```bash
# Test the optimized implementation
python test_orpheus_optimizations.py

# Deploy and test with Modal
modal run modal_app.py::test_orpheus_tts
```

### Expected Benchmarks on H100
- **Short text (10-20 chars):** TTFB <100ms, RTF <0.5x
- **Medium text (50-100 chars):** TTFB <150ms, RTF <0.8x  
- **Long text (200+ chars):** TTFB <200ms, RTF <1.2x

## 🔍 Key Technical Changes

### Model Loading Optimizations
```python
# H100-optimized model loading
self._model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # H100 optimized
    device_map="auto",
    use_cache=True,
    max_memory={0: "75GB"},  # Reserve H100 memory
    attn_implementation="flash_attention_2",  # Fastest attention
)
```

### Streaming Generation Config
```python
generation_config = {
    "max_new_tokens": 2048,  # Reduced for streaming
    "temperature": 0.8,      # Optimized for quality/speed
    "top_k": 40,            # H100 optimized
    "top_p": 0.9,           # Nucleus sampling
    "use_cache": True,      # Critical for streaming
    "output_scores": False, # Save memory
}
```

### Environment Variables
```bash
TORCH_CUDNN_V8_API_ENABLED=1
CUDA_LAUNCH_BLOCKING=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
```

## 🚨 Troubleshooting

### If performance is still slow:
1. **Check GPU:** Verify H100 is detected and utilized
2. **Memory:** Monitor GPU memory usage (should be 60-80%)
3. **Compilation:** Ensure torch.compile completed successfully
4. **Batch Size:** Verify SNAC batching is working
5. **Streaming:** Confirm tokens are yielded immediately

### Debug Commands
```python
# Check H100 detection
torch.cuda.get_device_properties(0)

# Monitor memory usage
torch.cuda.memory_allocated() / 1e9

# Verify compilation
print(f"Model compiled: {hasattr(model._model, '_orig_mod')}")
```

## 📈 Monitoring Performance

The optimized implementation includes detailed logging:
- Token generation speed (tokens/second)
- Memory utilization tracking
- TTFB measurements
- Real-time factor calculations
- H100-specific feature detection

## 🎯 Next Steps

1. **Deploy:** Test the optimized implementation on H100
2. **Monitor:** Check performance metrics in production
3. **Fine-tune:** Adjust batch sizes and timeouts based on results
4. **Scale:** Increase concurrency if performance targets are met

---

**Expected Result:** Orpheus TTS should now achieve **faster-than-real-time generation** (RTF < 1.0) with **sub-150ms TTFB** on H100 GPU, representing a **>15x performance improvement** over the previous implementation.