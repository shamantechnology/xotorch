"""
Inference engine for Cheetah C++ implementation.
"""
import os
import time
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional
import asyncio
import uuid
import socket
import json
import struct
import select

import numpy as np
import torch
import torchtune.generation as ttg

from xotorch.helpers import DEBUG
from xotorch.inference.shard import Shard
from xotorch.download.shard_download import ShardDownloader
from xotorch.inference.inference_engine import InferenceEngine
from xotorch.inference.tokenizers import _resolve_tokenizer
from xotorch.inference.torch.models.llm_utils import (
  load_model_config,
  ShardInferenceState,
  HF_PRECISION_DTYPE_TO_STR
)

TEMP = 0.6
TOP_K = 35
class CheetahInferenceEngine(InferenceEngine):
  """
  Inference engine for Cheetah C++ implementation.
  """

  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.cheetah_sock = None
    self.cheetah_header = {}
    self.request_id = None
    self.executor = ThreadPoolExecutor(max_workers=1)
    self.uuid = str(uuid.uuid4())
    self.model_path = None
    self.model_config = None
    self.state = None

    # device settings
    if os.environ.get("TORCH_DEVICE"):
      self.device = torch.device(os.environ["TORCH_DEVICE"])
    elif torch.cuda.is_available():
      self.device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
      self.device = torch.device("mps")
    else:
      self.device = torch.device("cpu")

    # rng setup for sampling
    self.rng = torch.Generator(device=self.device)
    self.rng.manual_seed(1234)

    # max sequence length
    self.max_seq_len = 1024

    # model token embed for sampling
    self.tok_embeddings = None
    
  
  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    """
    Encode the prompt into tokens.
    """
    
    await self.ensure_shard(shard)

    def encode_wrapper() -> np.ndarray:
      """
      Encode the tensors from prompt along with the
      initial input_pos and mask
      """
      tokens = self.tokenizer.encode(
          prompt,
          return_tensors="pt"
      )

      # move to proper device, default is CPU
      if tokens.device != self.device:
          tokens = tokens.to(device=self.device)
      
      if DEBUG >= 4:
          print("encoded_wrapper called")
          print(f"tokens: {tokens}")

      # Reset state
      self.state = ShardInferenceState(device=self.device)
      self.state.curr_pos = 0
      self.state.tokens = tokens

      _, tklng = tokens.size()

      total_response_length = tklng + self.max_seq_len

      if hasattr(self.tokenizer, "pad_id"):
        pad_id = self.tokenizer.pad_id
      elif hasattr(self.tokenizer, "pad_token_id"):
        if self.tokenizer.pad_token_id is not None:
          pad_id = self.tokenizer.pad_token_id
        else:
          pad_id = 0
      else:
        pad_id = 0

      padding_masks = tokens != pad_id
      if not padding_masks.all():
        padding_masks = torch.nn.functional.pad(
          padding_masks,
          (0, self.sharded_model.max_generated_tokens),
          value=True,
        )

        self.state.mask = ttg.get_causal_mask_from_padding_mask(padding_masks, target_seq_len=self.max_seq_len)

        self.state.input_pos = ttg.get_position_ids_from_padding_mask(padding_masks)
      else:
        self.state.mask = torch.tril(torch.ones(
          total_response_length,
          self.max_seq_len,
          dtype=torch.bool,
          device=self.device,
        )).unsqueeze(0)

        self.state.input_pos = torch.arange(0, total_response_length, device=self.device).unsqueeze(0)

      return tokens

    return await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(encode_wrapper),
    )

  async def sample(self, x: np.ndarray, temp=TEMP, top_k=TOP_K) -> np.ndarray:
      if DEBUG >= 4:
        print("sample called")
        print(f"x: {x}")
        print(f"temp: {temp}")
        print(f"top_k: {top_k}")
        print(self.device)

      logits = torch.tensor(x).to(self.device)

      def sample_wrapper():
        q = torch.empty(
          (
            logits.size(0),
            self.tok_embeddings.num_embeddings
          ), 
          device=logits.device).exponential_(1,
          generator=self.rng
        ).to(self.device)

        tokens = ttg.sample(
          logits.clone(),
          temperature=temp,
          top_k=top_k,
          q=q
        )
        
        if DEBUG >= 4:
          print(f"tokens: {tokens}")

        return tokens.numpy(force=True)

      return await asyncio.get_running_loop().run_in_executor(self.executor, functools.partial(sample_wrapper))

  async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
    if DEBUG >= 4:
      print("decode called")
      print(f"shard: {shard}")
      print(f"tokens: {tokens}")

    await self.ensure_shard(shard)

    return await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(self.tokenizer.decode, tokens.tolist()),
    )

  
  async def infer_tensor(
    self,
    request_id: str,
    shard: Shard,
    input_data: np.ndarray,
    inference_state: Optional[dict] = None
  ) -> Tuple[np.ndarray, Optional[dict]]:
    await self.ensure_shard(shard)

    # ensure shard
    if DEBUG >= 4:
      print("infer_tensor called")
      print(f"shard: {shard}")
      print(f"input_data: {input_data}")
      print(f"state {self.state}")

    if inference_state is not None and inference_state.get("tokens") is not None:
      self.state.from_dict(inference_state)

    self.request_id = request_id if not self.request_id else self.request_id

    hidden_state = None
    input_tensor = None
    if input_data.ndim == 3:
      hidden_state = torch.tensor(input_data).to(
        device=self.device,
        dtype=self.model_config["torch_dtype"]
      )
    elif input_data.ndim == 2:
      input_tensor = torch.tensor(input_data).to(
        device=self.device,
        dtype=torch.int
      )

    if input_tensor is not None:
      bsz, tklng = input_tensor.size()
    else:
      bsz, tklng = self.state.tokens.size()

    if self.state.tokens is not None:
      if input_data.ndim == 2 and input_tensor.size(-1) == 1:
        self.state.tokens = torch.cat([
          self.state.tokens.to(self.device),
          input_tensor.clone()
        ], dim=-1).to(self.device)
    else:
      self.state.tokens = input_tensor.clone()

    tokens = input_tensor.clone()

    input_pos = self.state.input_pos.clone()

    mask = self.state.mask.clone()

    curr_pos = self.state.curr_pos
    if curr_pos > 0:
      input_pos = input_pos[:, curr_pos].contiguous()
      mask = mask[:, curr_pos, None, :].contiguous()
    else:
      _, tklng = tokens.size()
      mask = mask[:, :tklng]
      input_pos = input_pos[:, :tklng].squeeze()
 
    return await self.run_cheetah(
      tokens=tokens,
      mask=mask,
      input_pos=input_pos,
      hidden_state=hidden_state
    )
  
  async def load_checkpoint(self, shard: Shard, path: str):
      """
      Load a checkpoint from the specified path.
      """
      pass

  async def save_checkpoint(self, shard: Shard, path: str):
      """
      Save the current state to a checkpoint file.
      """
      pass

  async def save_session(self, key, value):
      self.session[key] = value

  async def clear_session(self):
      self.session.clear()

  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, inference_state: Optional[dict] = None) -> Tuple[np.ndarray, Optional[dict]]:
    tokens = await self.encode(shard, prompt)
    x = tokens.reshape(1, -1)
    output_data, inference_state = await self.infer_tensor(request_id, shard, x, inference_state)

    return output_data, inference_state

  async def ensure_shard(self, shard: Shard):
    if DEBUG >= 4:
      print("shard ensured\n")
      print(f"shard: {shard}")
      print(f"class shard: {self.shard}")
      print(f"uuid: {self.uuid}")

    if self.shard == shard:
      return

    self.shard = shard
    self.state = ShardInferenceState()

    self.model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
    self.model_config = load_model_config(self.model_path/"config.json")

    if os.environ.get("XOTORCH_MAX_SEQ_LEN"):
      self.max_seq_len = int(os.environ["XOTORCH_MAX_SEQ_LEN"])
    else:
      self.max_seq_len = self.model_config.get("max_seq_len", 1024)

    # self.tokenizer = await _resolve_tokenizer(model_path)
    self.tokenizer = await _resolve_tokenizer(self.model_path)

    # utilize tok embed for sampling
    self.tok_embeddings = torch.nn.Embedding(
      num_embeddings=self.model_config["vocab_size"],
      embedding_dim=self.model_config["embed_dim"],
      device=self.device,
      dtype=self.model_config["torch_dtype"]
    )

    self.cheetah_header = {
      "node_id": self.uuid,
      "model": self.shard.model_id,
      "model_path": str(self.model_path),
      "layer_start": shard.start_layer,
      "layer_end": shard.end_layer,
      "layer_total": shard.n_layers,
      "has_hidden_state": False
    }

  async def run_cheetah(
    self,
    tokens: torch.Tensor,
    input_pos: torch.Tensor,
    mask: torch.Tensor,
    hidden_state: Optional[torch.Tensor] = None
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Send model data to model running on cheetah
    recieved model output (hidden values or logits)
    """
    # setup connection
    self.cheetah_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    self.cheetah_sock.setblocking(False)
    try:
      self.cheetah_sock.connect("/tmp/cheetah_infra")
    except Exception as err:
      print(f"/tmp/cheetah_infra not found\n{err}")
      raise

    if DEBUG >= 4:
      print("Connected to cheetah service socket @ /tmp/cheetah_infra")

    # get asyncio loop
    loop = asyncio.get_running_loop()

    # setup header
    self.cheetah_header["session_id"] = self.request_id
    if hidden_state is None:
      # imi is input_ids, mask, input_pos
      self.cheetah_header["input_ids_shape"] = list(tokens.shape)
      self.cheetah_header["mask_shape"] = list(mask.shape)
      self.cheetah_header["input_pos_shape"] = list(input_pos.shape)
      self.cheetah_header["dtype_input_ids"] = HF_PRECISION_DTYPE_TO_STR[tokens.dtype]
      self.cheetah_header["dtype_mask"] = HF_PRECISION_DTYPE_TO_STR[mask.dtype]
      self.cheetah_header["dtype_input_pos"] = HF_PRECISION_DTYPE_TO_STR[input_pos.dtype]
    else:
      self.cheetah_header["hidden_state_dtype"] =  HF_PRECISION_DTYPE_TO_STR[hidden_state.dtype]
      self.cheetah_header["hidden_state_shape"] = list(hidden_state.shape)
      self.cheetah_header["has_hidden_state"] = True

    # setup payload
    header_bytes = json.dumps(self.cheetah_header).encode("utf-8")
    if hidden_state is None:
      payload = tokens.numpy(force=True).tobytes() + \
                mask.numpy(force=True).astype(np.uint8).tobytes() + \
                input_pos.numpy(force=True).tobytes()
    
      await loop.sock_sendall(self.cheetah_sock, struct.pack("!I", len(header_bytes)))
      await loop.sock_sendall(self.cheetah_sock, header_bytes)
      await loop.sock_sendall(self.cheetah_sock, payload)
    else:
      if hidden_state.is_floating_point():    
        hidden_payload = hidden_state.float().numpy(force=True).tobytes()
        await loop.sock_sendall(self.cheetah_sock, struct.pack("!f", len(header_bytes)))
        await loop.sock_sendall(self.cheetah_sock, header_bytes)
        await loop.sock_sendall(self.cheetah_sock, hidden_payload)
      else:
        hidden_payload = hidden_state.numpy(force=True).tobytes()
        await loop.sock_sendall(self.cheetah_sock, struct.pack("!I", len(header_bytes)))
        await loop.sock_sendall(self.cheetah_sock, header_bytes)
        await loop.sock_sendall(self.cheetah_sock, hidden_payload)

    # wait for response
    if DEBUG >= 4:
      print("Waiting for Cheetah response...")
    
    raw_len = await loop.sock_recv(self.cheetah_sock, 4)
    if not raw_len:
      raise ConnectionError("Did not receive header length")
    
    if DEBUG >= 4:
      print("Cheetah response received")
    
    header_len = struct.unpack("!I", raw_len)[0]

    header_data = await loop.sock_recv(self.cheetah_sock, header_len)
    header = json.loads(header_data.decode("utf-8"))
    
    if DEBUG >= 4:
      print(f"RECV Cheetah header: {header}")

    shape = header["shape"]
    dtype = header["dtype"]

    if dtype == "bfloat16":
        bytes_per_element = 2
        np_dtype = np.uint16
    elif dtype in ("float32", "float"):
        bytes_per_element = 4
        np_dtype = np.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    expected_bytes = np.prod(shape) * bytes_per_element
    data = bytearray()
    while len(data) < expected_bytes:
      chunk = await loop.sock_recv(self.cheetah_sock, expected_bytes - len(data))
      if not chunk:
        raise ConnectionError("Incomplete tensor data received")
      data += chunk

    if dtype == "bfloat16":
      bfloat16_data = np.frombuffer(data, dtype=np.uint16).reshape(shape)
      float32_bits = bfloat16_data.astype(np.uint32) << 16
      out_tensor = float32_bits.view(np.float32)
    else:
      out_tensor = np.frombuffer(data, dtype=np_dtype).reshape(shape)

    if hidden_state is not None:
      return (
        out_tensor,
        self.state.to_dict(),
      )

    if self.state.curr_pos == 0:
      self.state.curr_pos = self.state.tokens.size(-1)
    else:
      self.state.curr_pos += 1

    return (
      out_tensor[:, -1],
      self.state.to_dict()
    )
