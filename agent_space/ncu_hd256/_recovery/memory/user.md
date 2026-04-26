---
name: User profile
description: Kernel engineer at Together.ai working on FlashAttention-4 (FA4) CuTeDSL kernels for NVIDIA Blackwell.
type: user
originSessionId: 5ce87721-bf30-40aa-8632-d0b628cc1cc5
---
User: johnson@together.ai. Works on FlashAttention-4 (FA4) kernels in `/workspace/flash-attention/flash_attn/cute/`, written in CuTeDSL (NVIDIA CUTLASS DSL), targeting Blackwell SM100 (B200). Also has `/workspace/cutlass/` checked out alongside.

Comfortable with low-level GPU specifics: PDL (Programmatic Dependent Launch), `griddepcontrol_wait`, TMA, mbarrier, 2CTA cluster instructions, CUDA streams/events, TVM FFI env-stream injection. Reads PTX/SASS, profiles with nsys.

Style: terse, direct. Prefers concise responses with concrete numbers and short verdicts over long explanations. Does not need GPU/CUDA concepts re-explained.

Working environment: 8× B200 node, no sudo needed for `nvidia-smi -lgc` clock locking.
