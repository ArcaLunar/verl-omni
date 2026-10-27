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
"""Byte-Prefix Marginalization (BPM) for cross-tokenizer on-policy distillation.

Implements the target construction of "Cross-Tokenizer On-Policy Distillation via
Byte-Prefix Marginalization" (arXiv:2607.22334). The teacher's next-token distribution
is re-expressed on the student vocabulary through a shared byte space, so a teacher with
a different tokenizer can supervise every student response position.

The modules here are torch-free except :mod:`encoding`, and run on CPU inside the agent
loop.
"""
