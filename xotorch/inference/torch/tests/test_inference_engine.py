"""
Test inference engine and model sharding
"""
import pytest
import asyncio
import time

import numpy as np

from xotorch.inference.shard import Shard
from xotorch.inference.torch.sharded_inference_engine import TorchDynamicShardInferenceEngine
from xotorch.download.new_shard_download import new_shard_downloader
from xotorch.download.shard_download import ShardDownloader

@pytest.mark.asyncio
async def test_inference_engine():
  prompt = "Tell me a haiku in 10 words or less."

  shard = Shard(
    model_id="unsloth/Llama-3.2-1B-Instruct",
    start_layer=0,
    end_layer=8,
    n_layers=16
  )

  shard_2 = Shard(
    model_id="unsloth/Llama-3.2-1B-Instruct",
    start_layer=9,
    end_layer=15,
    n_layers= 16
  )

  shard_downloader = new_shard_downloader()
  inference_engine = TorchDynamicShardInferenceEngine(shard_downloader)

  current_time = time.time()

  output_1 = await inference_engine.infer_prompt("test_id", shard, prompt)
  print("\n------------inference_engine.infer_prompt output---------------\n")
  print(output_1[0].shape)
  print("\n---------------------------\n")

  elapsed_time = time.time() - current_time
  tokens_per_second = len(output_1) / elapsed_time
  print(f"Time taken: {elapsed_time:.2f} seconds")
  print(f"Tokens per second: {tokens_per_second:.2f}")  

  assert isinstance(output_1[0], np.ndarray), "Output should be numpy array"

  # output_2 = await inference_engine.infer_tensor("test_id", shard, output_1) 
  # print("\n------------inference_engine.infer_tensor output---------------\n")
  # print(output_2)
  # print("\n---------------------------\n")

  # assert isinstance(output_2, np.ndarray), "Output should be numpy array" 

if __name__ == '__main__':
  try:
    print("\n\n -------- TEST llama-3.2-1b -------- \n\n")
    asyncio.run(test_inference_engine())
  except Exception as err:
    print(f"\n!!!! TEST FAILED \n{err}\n")


