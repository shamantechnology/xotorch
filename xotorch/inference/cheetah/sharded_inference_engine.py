"""
Inference engine for Cheetah C++ implementation.
"""
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional
import asyncio
import uuid
import numpy as np
import torch
import torchtune.generation as ttg
import socket
import json
import struct
import select

from xotorch.helpers import DEBUG
from xotorch.inference.shard import Shard
from xotorch.download.shard_download import ShardDownloader
from xotorch.inference.inference_engine import InferenceEngine
from xotorch.inference.torch.models.llm_utils import (
  load_model_config,
  load_model_weights_torchtune,
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
      max_seq_len = self.model_config["max_seq_len"]
      total_response_length = tklng + max_seq_len

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

        self.state.mask = ttg.get_causal_mask_from_padding_mask(padding_masks, target_seq_len=max_seq_len)

        self.state.input_pos = ttg.get_position_ids_from_padding_mask(padding_masks)
      else:
        self.state.mask = torch.tril(torch.ones(
          total_response_length,
          max_seq_len,
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
        tokens = ttg.sample(logits.clone(), temperature=temp, top_k=top_k)
        
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
      self.setup_cache(
        bsz,
        tklng + self.sharded_model.max_generated_tokens
      )
    else:
      bsz, tklng = self.state.tokens.size()
      self.setup_cache(
        bsz,
        tklng + self.sharded_model.max_generated_tokens
      )

    if self.state.tokens is not None:
      if input_data.ndim == 2 and input_tensor.size(-1) == 1:
        self.state.tokens = torch.cat([
          self.state.tokens.to(self.device),
          input_tensor.clone()
        ], dim=-1).to(self.device)
    else:
      self.state.tokens = input_tensor.clone()

    tokens = self.state.tokens.clone()

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

    self.cheetah_header["input_id_shape"] = list(tokens.shape)
    self.cheetah_header["mask_shape"] = list(mask.shape)
    self.cheetah_header["input_pos_shape"] = list(input_pos.shape)

    def infer_wrapper():
      if DEBUG >= 4:
        print(f"infer_wrapper called")
        print(f"self.state.tokens: {self.state.tokens}")
        print(f"hidden_state: {hidden_state}")

      if hidden_state is not None:
        model_hs, model_logits = self.run_cheetah(
          tokens=tokens,
          hidden_state=hidden_state,
          input_pos=input_pos,
          mask=mask,
        )
      else:
        model_hs, model_logits = self.run_cheetah(
          tokens=tokens,
          input_pos=input_pos,
          mask=mask,
        )

      if model_hs is not None:
        return (
          model_hs,
          self.state.to_dict(),
        )
      
      if self.state.curr_pos == 0:
        self.state.curr_pos = self.state.tokens.size(-1)
      else:
        self.state.curr_pos += 1

      return (
        model_logits[:, -1],
        self.state.to_dict(),
      )

    return await asyncio.get_running_loop().run_in_executor(self.executor, infer_wrapper)
  
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

    try:
      self.cheetah_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
      self.cheetah_sock.connect("/run/cheetah_infra")
    except Exception as err:
      print(f"/run/cheetah_infra not found\n{err}")
      raise

    self.model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
    self.model_config = load_model_config(self.model_path/"config.json")

    self.cheetah_header = {
      "node_id": self.uuid,
      "model": self.shard.model_id,
      "model_path": str(self.model_path),
      "layer_start": shard.start_layer,
      "layer_end": shard.end_layer,
      "layer_total": shard.n_layers,
      "dtype": "int64",
      "input_id_shape": [],
      "mask_shape": [],
      "input_pos_shape": [],
      "has_hidden_state": False
    }

  def run_cheetah(
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
    # setup payload
    if hidden_state is None:
      payload = tokens.numpy(force=True).tobytes() + \
                mask.numpy(force=True).tobytes() + \
                input_pos.numpy(force=True).tobytes()
    
      # send header and payload
      header_bytes = json.dumps(self.cheetah_header).encode("utf-8")
      self.cheetah_sock.sendall(struct.pack("!I", len(header_bytes)))
      self.cheetah_sock.sendall(header_bytes)
      self.cheetah_sock.sendall(payload)
    else:
      self.hidden_state_header = {
        "node_id": self.uuid,
        "model": self.shard.model_id,
        "model_path": str(self.model_path),
        "layer_start": self.shard.start_layer,
        "layer_end": self.shard.end_layer,
        "layer_total": self.shard.n_layers,
        "dtype": HF_PRECISION_DTYPE_TO_STR[hidden_state.dtype],
        "hidden_state_shape": list(hidden_state.shape)
      }

      tmip_payload = tokens.numpy(force=True).tobytes() + \
        mask.numpy(force=True).tobytes() + \
        input_pos.numpy(force=True).tobytes()

      self.cheetah_header["has_hidden_state"] = True

      header_bytes = json.dumps(self.cheetah_header).encode("utf-8")
      self.cheetah_sock.sendall(struct.pack("!I", len(header_bytes)))
      self.cheetah_sock.sendall(header_bytes)
      self.cheetah_sock.sendall(tmip_payload)

      header_bytes = json.dumps(self.hidden_state_header).encode("utf-8")
      if hidden_state.is_floating_point():
        # send int payload then float payload       
        hidden_payload = hidden_state.float().numpy(force=True).tobytes()
        self.cheetah_sock.sendall(struct.pack("!f", len(header_bytes)))
        self.cheetah_sock.sendall(header_bytes)
        self.cheetah_sock.sendall(hidden_payload)
      else:
        hidden_payload = hidden_state.numpy(force=True).tobytes()
        self.cheetah_sock.sendall(struct.pack("!I", len(header_bytes)))
        self.cheetah_sock.sendall(header_bytes)
        self.cheetah_sock.sendall(hidden_payload)

    # wait for response
    readable_sockets, _, _ = select.select([self.cheetah_sock], [], [], 0)
    if not readable_sockets:
      raise ConnectionError("No readable sockets available")
    
    print("Waiting for cheetah response...")
    while self.cheetah_sock not in readable_sockets:
      readable_sockets, _, _ = select.select([self.cheetah_sock], [], [], 0)
    
    print("Cheetah response received")
    raw_len = self.cheetah_sock.recv(4)
    if not raw_len:
        raise ConnectionError("Did not receive header length")
    header_len = struct.unpack("!I", raw_len)[0]

    header_data = self.cheetah_sock.recv(header_len)
    header = json.loads(header_data.decode("utf-8"))
    
    if DEBUG >= 4:
      print(f"header: {header}")

    shape = header["shape"]
    dtype = header["dtype"]

    if dtype in ("float32", "float"):
        np_dtype = np.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    expected_bytes = np.prod(shape) * np.dtype(np_dtype).itemsize
    data = b""
    while len(data) < expected_bytes:
        chunk = self.cheetah_sock.recv(expected_bytes - len(data))
        if not chunk:
            raise ConnectionError("Incomplete tensor data received")
        data += chunk

    out_tensor = np.frombuffer(data, dtype=np_dtype).reshape(shape)
    if hidden_state is not None:
      return out_tensor, None

    return None, out_tensor
