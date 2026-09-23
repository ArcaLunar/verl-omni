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
import pickle

import torch
from tensordict import TensorDict
from verl import DataProto

import verl_omni.utils.dataproto_serialization as dataproto_patch


def _roundtrip(tensor, batch_size):
    """pickle round-trip of a ``DataProto`` with a single ``responses`` tensor."""
    data = DataProto(batch=TensorDict({"responses": tensor}, batch_size=batch_size))
    restored = pickle.loads(pickle.dumps(data))
    return restored.batch["responses"]


def _apply_patch(monkeypatch):
    # Reset the module-level idempotency flag so each test observes a fresh run.
    monkeypatch.setattr(dataproto_patch, "_patched", False)
    dataproto_patch.patch_dataproto_serialization()


def test_empty_tensor_roundtrip_does_not_raise(monkeypatch):
    monkeypatch.setenv("VERL_DATAPROTO_SERIALIZATION_METHOD", "numpy")
    _apply_patch(monkeypatch)

    cases = [
        (torch.empty(1, 0, dtype=torch.uint8), (1,)),  # single failed sample
        (torch.empty(32, 0, dtype=torch.uint8), (32,)),  # all-failed chunk [B, 0]
        (torch.empty(4, 0, 3, 256, 384, dtype=torch.uint8), (4,)),
        (torch.empty(2, 0, dtype=torch.bfloat16), (2,)),
    ]
    for tensor, batch_size in cases:
        out = _roundtrip(tensor, batch_size)
        assert out.shape == tensor.shape, (out.shape, tensor.shape)
        assert out.dtype == tensor.dtype, (out.dtype, tensor.dtype)


def test_non_empty_tensor_still_roundtrips_verbatim(monkeypatch):
    monkeypatch.setenv("VERL_DATAPROTO_SERIALIZATION_METHOD", "numpy")
    _apply_patch(monkeypatch)

    tensor = torch.stack([torch.tensor([1, 2, 3], dtype=torch.uint8)] * 2)
    out = _roundtrip(tensor, (2,))
    assert torch.equal(out, tensor)


def test_patch_is_noop_without_numpy_method(monkeypatch):
    import verl.protocol as verl_protocol

    monkeypatch.delenv("VERL_DATAPROTO_SERIALIZATION_METHOD", raising=False)
    original = verl_protocol.deserialize_single_tensor

    _apply_patch(monkeypatch)

    assert verl_protocol.deserialize_single_tensor is original
