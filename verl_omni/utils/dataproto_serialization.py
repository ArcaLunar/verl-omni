# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Empty-safe round-trip for verl's numpy ``DataProto`` serialization.

With ``VERL_DATAPROTO_SERIALIZATION_METHOD=numpy``, verl's
``deserialize_single_tensor`` (``verl/protocol.py``) reconstructs every tensor
via ``torch.frombuffer``. A zero-length tensor (e.g. a diffusion request that
finished with no image) serializes to a 0-byte buffer, and ``torch.frombuffer``
raises ``ValueError: both buffer length (0) and count (-1) must not be 0``.
Ray surfaces that as a ``RaySystemError``, turning one bad sample into a crash.
This patch reconstructs an empty tensor from the already-encoded shape/dtype
instead, so zero-length tensors round-trip identically to non-empty ones.
"""

import os

import torch

_patched = False


def patch_dataproto_serialization() -> None:
    """Make verl's numpy DataProto serialization round-trip zero-length tensors.

    Idempotent. Applied only when the numpy serialization method is active.
    ``deserialize_single_tensor`` is referenced as a module global inside
    ``verl/protocol.py`` at call time, so patching the attribute covers every
    (de)serialization path; no other module imports it by name.
    """
    if os.getenv("VERL_DATAPROTO_SERIALIZATION_METHOD") != "numpy":
        return

    global _patched
    if _patched:
        return

    import verl.protocol as verl_protocol

    original = verl_protocol.deserialize_single_tensor

    def deserialize_single_tensor(arr):
        dtype, shape, data = arr
        if not bytearray(data):
            torch_dtype = getattr(torch, dtype)
            return torch.empty(shape, dtype=torch_dtype)
        return original(arr)

    verl_protocol.deserialize_single_tensor = deserialize_single_tensor
    _patched = True
