from typing import Any
from pydantic import BaseModel
import psutil
import torch
import platform

import logging
from datetime import datetime
from pathlib import Path

from xotorch import DEBUG
from xotorch.helpers import get_mac_system_info, subprocess_pool
# 
logging.basicConfig(
    filename=f"run_{datetime.now().strftime('%Y_%m_%d')}.log", # move to log folder
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"
)

TFLOPS = 1.00
class DeviceFlops(BaseModel):
  # units of TFLOPS
  fp32: float
  fp16: float
  int8: float

  def __str__(self):
    return f"fp32: {self.fp32 / TFLOPS:.2f} TFLOPS, fp16: {self.fp16 / TFLOPS:.2f} TFLOPS, int8: {self.int8 / TFLOPS:.2f} TFLOPS"

  def to_dict(self):
    return self.model_dump()


class DeviceCapabilities(BaseModel):
  model: str
  chip: str
  memory: int
  flops: DeviceFlops

  def __str__(self):
    return f"Model: {self.model}. Chip: {self.chip}. Memory: {self.memory}MB. Flops: {self.flops}"

  def model_post_init(self, __context: Any) -> None:
    if isinstance(self.flops, dict):
      self.flops = DeviceFlops(**self.flops)

  def to_dict(self):
    return {"model": self.model, "chip": self.chip, "memory": self.memory, "flops": self.flops.to_dict()}


UNKNOWN_DEVICE_CAPABILITIES = DeviceCapabilities(model="Unknown Model", chip="Unknown Chip", memory=0, flops=DeviceFlops(fp32=0, fp16=0, int8=0))

CHIP_FLOPS = {
  # Source: https://www.cpu-monkey.com
  # Note: currently no distinction between variants of M3 Max and M3 Pro, we pick the lower one to be conservative
  ### M chips
  "Apple M1": DeviceFlops(fp32=2.29*TFLOPS, fp16=4.58*TFLOPS, int8=9.16*TFLOPS),
  "Apple M1 Pro": DeviceFlops(fp32=5.30*TFLOPS, fp16=10.60*TFLOPS, int8=21.20*TFLOPS),
  "Apple M1 Max": DeviceFlops(fp32=10.60*TFLOPS, fp16=21.20*TFLOPS, int8=42.40*TFLOPS),
  "Apple M1 Ultra": DeviceFlops(fp32=21.20*TFLOPS, fp16=42.40*TFLOPS, int8=84.80*TFLOPS),
  "Apple M2": DeviceFlops(fp32=3.55*TFLOPS, fp16=7.10*TFLOPS, int8=14.20*TFLOPS),
  "Apple M2 Pro": DeviceFlops(fp32=5.68*TFLOPS, fp16=11.36*TFLOPS, int8=22.72*TFLOPS),
  "Apple M2 Max": DeviceFlops(fp32=13.49*TFLOPS, fp16=26.98*TFLOPS, int8=53.96*TFLOPS),
  "Apple M2 Ultra": DeviceFlops(fp32=26.98*TFLOPS, fp16=53.96*TFLOPS, int8=107.92*TFLOPS),
  "Apple M3": DeviceFlops(fp32=3.55*TFLOPS, fp16=7.10*TFLOPS, int8=14.20*TFLOPS),
  "Apple M3 Pro": DeviceFlops(fp32=4.97*TFLOPS, fp16=9.94*TFLOPS, int8=19.88*TFLOPS),
  "Apple M3 Max": DeviceFlops(fp32=14.20*TFLOPS, fp16=28.40*TFLOPS, int8=56.80*TFLOPS),
  "Apple M3 Ultra": DeviceFlops(fp32=54.26*TFLOPS, fp16=108.52*TFLOPS, int8=217.04*TFLOPS),
  "Apple M4": DeviceFlops(fp32=4.26*TFLOPS, fp16=8.52*TFLOPS, int8=17.04*TFLOPS),
  "Apple M4 Pro": DeviceFlops(fp32=5.72*TFLOPS, fp16=11.44*TFLOPS, int8=22.88*TFLOPS),
  "Apple M4 Max": DeviceFlops(fp32=18.03*TFLOPS, fp16=36.07*TFLOPS, int8=72.14*TFLOPS),
  ### A chips
  "Apple A13 Bionic": DeviceFlops(fp32=0.69*TFLOPS, fp16=1.38*TFLOPS, int8=2.76*TFLOPS),
  "Apple A14 Bionic": DeviceFlops(fp32=0.75*TFLOPS, fp16=1.50*TFLOPS, int8=3.00*TFLOPS),
  "Apple A15 Bionic": DeviceFlops(fp32=1.37*TFLOPS, fp16=2.74*TFLOPS, int8=5.48*TFLOPS),
  "Apple A16 Bionic": DeviceFlops(fp32=1.79*TFLOPS, fp16=3.58*TFLOPS, int8=7.16*TFLOPS),
  "Apple A17 Pro": DeviceFlops(fp32=2.15*TFLOPS, fp16=4.30*TFLOPS, int8=8.60*TFLOPS),
  ### NVIDIA GPUs
  # RTX 40 series
  "NVIDIA GEFORCE RTX 4090": DeviceFlops(fp32=82.58*TFLOPS, fp16=165.16*TFLOPS, int8=330.32*TFLOPS),
  "NVIDIA GEFORCE RTX 4090 D": DeviceFlops(fp32=82.58*TFLOPS, fp16=165.16*TFLOPS, int8=330.32*TFLOPS),
  "NVIDIA GEFORCE RTX 4080": DeviceFlops(fp32=48.74*TFLOPS, fp16=97.48*TFLOPS, int8=194.96*TFLOPS),
  "NVIDIA GEFORCE RTX 4080 SUPER": DeviceFlops(fp32=52.0*TFLOPS, fp16=104.0*TFLOPS, int8=208.0*TFLOPS),
  "NVIDIA GEFORCE RTX 4070 TI SUPER": DeviceFlops(fp32=40.0*TFLOPS, fp16=80.0*TFLOPS, int8=160.0*TFLOPS),
  "NVIDIA GEFORCE RTX 4070 TI": DeviceFlops(fp32=39.43*TFLOPS, fp16=78.86*TFLOPS, int8=157.72*TFLOPS),
  "NVIDIA GEFORCE RTX 4070 SUPER": DeviceFlops(fp32=30.0*TFLOPS, fp16=60.0*TFLOPS, int8=120.0*TFLOPS),
  "NVIDIA GEFORCE RTX 4070": DeviceFlops(fp32=29.0*TFLOPS, fp16=58.0*TFLOPS, int8=116.0*TFLOPS),
  "NVIDIA GEFORCE RTX 4060 TI 16GB": DeviceFlops(fp32=22.0*TFLOPS, fp16=44.0*TFLOPS, int8=88.0*TFLOPS),
  "NVIDIA GEFORCE RTX 4060 TI": DeviceFlops(fp32=22.0*TFLOPS, fp16=44.0*TFLOPS, int8=88.0*TFLOPS),
  # RTX 30 series
  "NVIDIA GEFORCE RTX 3050": DeviceFlops(fp32=9.11*TFLOPS, fp16=18.22*TFLOPS, int8=36.44*TFLOPS),
  "NVIDIA GEFORCE RTX 3060": DeviceFlops(fp32=13.0*TFLOPS, fp16=26.0*TFLOPS, int8=52.0*TFLOPS),
  "NVIDIA GEFORCE RTX 3060 TI": DeviceFlops(fp32=16.2*TFLOPS, fp16=32.4*TFLOPS, int8=64.8*TFLOPS),
  "NVIDIA GEFORCE RTX 3070": DeviceFlops(fp32=20.3*TFLOPS, fp16=40.6*TFLOPS, int8=81.2*TFLOPS),
  "NVIDIA GEFORCE RTX 3070 TI": DeviceFlops(fp32=21.8*TFLOPS, fp16=43.6*TFLOPS, int8=87.2*TFLOPS),
  "NVIDIA GEFORCE RTX 3080 (10 GB)": DeviceFlops(fp32=29.8*TFLOPS, fp16=59.6*TFLOPS, int8=119.2*TFLOPS),
  "NVIDIA GEFORCE RTX 3080 (12 GB)": DeviceFlops(fp32=30.6*TFLOPS, fp16=61.2*TFLOPS, int8=122.4*TFLOPS),
  "NVIDIA GEFORCE RTX 3080 TI": DeviceFlops(fp32=34.1*TFLOPS, fp16=68.2*TFLOPS, int8=136.4*TFLOPS),
  "NVIDIA GEFORCE RTX 3090": DeviceFlops(fp32=35.6*TFLOPS, fp16=71.2*TFLOPS, int8=142.4*TFLOPS),
  "NVIDIA GEFORCE RTX 3090 TI": DeviceFlops(fp32=40.0*TFLOPS, fp16=80.0*TFLOPS, int8=160.0*TFLOPS),
  # RTX 20 series
  "NVIDIA GEFORCE RTX 2060": DeviceFlops(fp32=6.45*TFLOPS, fp16=12.9*TFLOPS, int8=25.8*TFLOPS),
  "NVIDIA GEFORCE RTX 2060 SUPER": DeviceFlops(fp32=7.2*TFLOPS, fp16=14.4*TFLOPS, int8=28.8*TFLOPS),
  "NVIDIA GEFORCE RTX 2070": DeviceFlops(fp32=7.46*TFLOPS, fp16=14.93*TFLOPS, int8=29.86*TFLOPS),
  "NVIDIA GEFORCE RTX 2070 SUPER": DeviceFlops(fp32=9.06*TFLOPS, fp16=18.12*TFLOPS, int8=36.24*TFLOPS),
  "NVIDIA GEFORCE RTX 2080": DeviceFlops(fp32=10.07*TFLOPS, fp16=20.14*TFLOPS, int8=40.28*TFLOPS),
  "NVIDIA GEFORCE RTX 2080 TI": DeviceFlops(fp32=13.45*TFLOPS, fp16=26.9*TFLOPS, int8=40.28*TFLOPS),
  "NVIDIA GEFORCE RTX 2080 SUPER": DeviceFlops(fp32=11.15*TFLOPS, fp16=22.30*TFLOPS, int8=44.60*TFLOPS),
  "NVIDIA TITAN RTX": DeviceFlops(fp32=16.31*TFLOPS, fp16=32.62*TFLOPS, int8=65.24*TFLOPS),
  # GTX 10 series
  "NVIDIA GEFORCE GTX 1050 TI": DeviceFlops(fp32=2.0*TFLOPS, fp16=4.0*TFLOPS, int8=8.0*TFLOPS),
  "NVIDIA GEFORCE GTX 1070": DeviceFlops(fp32=6.463*TFLOPS, fp16=0.101*TFLOPS, int8=25.852*TFLOPS),
  "NVIDIA GEFORCE GTX 1080": DeviceFlops(fp32=8.873*TFLOPS, fp16=0.138*TFLOPS, int8=35.492*TFLOPS),
  "NVIDIA GEFORCE GTX 1080 TI": DeviceFlops(fp32=11.34*TFLOPS, fp16=0.177*TFLOPS, int8=45.36*TFLOPS),
  # GTX 16 series
  "NVIDIA GeForce GTX 1660 TI": DeviceFlops(fp32=4.8*TFLOPS, fp16=9.6*TFLOPS, int8=19.2*TFLOPS),
  # QUADRO RTX Ampere series
  "NVIDIA RTX A2000": DeviceFlops(fp32=7.99*TFLOPS, fp16=7.99*TFLOPS, int8=31.91*TFLOPS),
  "NVIDIA RTX A4000": DeviceFlops(fp32=19.17*TFLOPS, fp16=19.17*TFLOPS, int8=76.68*TFLOPS),
  "NVIDIA RTX A4500": DeviceFlops(fp32=23.65*TFLOPS, fp16=23.65*TFLOPS, int8=94.6*TFLOPS),
  "NVIDIA RTX A5000": DeviceFlops(fp32=27.8*TFLOPS, fp16=27.8*TFLOPS, int8=111.2*TFLOPS),
  "NVIDIA RTX A6000": DeviceFlops(fp32=38.71*TFLOPS, fp16=38.71*TFLOPS, int8=154.84*TFLOPS),
  # NVIDIA Ada Lovelace Architecture-Based
  "NVIDIA RTX 4000 ADA GENERATION": DeviceFlops(fp32=26.7*TFLOPS, fp16=26.7*TFLOPS, int8=258.0*TFLOPS),
  # Common Server GPUs
  "NVIDIA A40 48GB PCIE": DeviceFlops(fp32=37.4*TFLOPS, fp16=149.7*TFLOPS, int8=299.3*TFLOPS),
  "NVIDIA A100 40GB PCIE": DeviceFlops(fp32=19.5*TFLOPS, fp16=312.0*TFLOPS, int8=624.0*TFLOPS),
  "NVIDIA A800 40GB PCIE": DeviceFlops(fp32=19.5*TFLOPS, fp16=312.0*TFLOPS, int8=624.0*TFLOPS),
  "NVIDIA A100 80GB PCIE": DeviceFlops(fp32=19.5*TFLOPS, fp16=312.0*TFLOPS, int8=624.0*TFLOPS),
  "NVIDIA A800 80GB PCIE": DeviceFlops(fp32=19.5*TFLOPS, fp16=312.0*TFLOPS, int8=624.0*TFLOPS),
  "NVIDIA A100 80GB SXM": DeviceFlops(fp32=19.5*TFLOPS, fp16=312.0*TFLOPS, int8=624.0*TFLOPS),
  "NVIDIA A800 80GB SXM": DeviceFlops(fp32=19.5*TFLOPS, fp16=312.0*TFLOPS, int8=624.0*TFLOPS),
  "NVIDIA T1000 8GB": DeviceFlops(fp32=2.5 * TFLOPS, fp16=5.0 * TFLOPS, int8=10.0 * TFLOPS),
  "QUADRO M2000": DeviceFlops(fp32=0.5 * TFLOPS, fp16=1.0 * TFLOPS, int8=2.0 * TFLOPS),
  "QUADRO P400": DeviceFlops(fp32=0.641 * TFLOPS, fp16=1.282 * TFLOPS, int8=2.564 * TFLOPS),
  "NVIDIA A10": DeviceFlops(fp32=31.2 * TFLOPS, fp16=62.5 * TFLOPS, int8=2.5 * TFLOPS),
  # ... add more devices if needed ...
  ### AMD GPUs
  # RX 6000 series
  "AMD Radeon RX 6900 XT": DeviceFlops(fp32=23.04*TFLOPS, fp16=46.08*TFLOPS, int8=92.16*TFLOPS),
  "AMD Radeon RX 6800 XT": DeviceFlops(fp32=20.74*TFLOPS, fp16=41.48*TFLOPS, int8=82.96*TFLOPS),
  "AMD Radeon RX 6800": DeviceFlops(fp32=16.17*TFLOPS, fp16=32.34*TFLOPS, int8=64.68*TFLOPS),
  "AMD Radeon RX 6700 XT": DeviceFlops(fp32=13.21*TFLOPS, fp16=26.42*TFLOPS, int8=52.84*TFLOPS),
  "AMD Radeon RX 6700": DeviceFlops(fp32=11.4*TFLOPS, fp16=22.8*TFLOPS, int8=45.6*TFLOPS),
  "AMD Radeon RX 6600 XT": DeviceFlops(fp32=10.6*TFLOPS, fp16=21.2*TFLOPS, int8=42.4*TFLOPS),
  "AMD Radeon RX 6600": DeviceFlops(fp32=8.93*TFLOPS, fp16=17.86*TFLOPS, int8=35.72*TFLOPS),
  "AMD Radeon RX 6500 XT": DeviceFlops(fp32=5.77*TFLOPS, fp16=11.54*TFLOPS, int8=23.08*TFLOPS),
  "AMD Radeon RX 6400": DeviceFlops(fp32=3.57*TFLOPS, fp16=7.14*TFLOPS, int8=14.28*TFLOPS),
  # RX 7000 series
  "AMD Radeon RX 7900 XTX": DeviceFlops(fp32=61.4*TFLOPS, fp16=122.8*TFLOPS, int8=245.6*TFLOPS),
  "AMD Radeon RX 7900 XT": DeviceFlops(fp32=53.4*TFLOPS, fp16=106.8*TFLOPS, int8=213.6*TFLOPS),
  "AMD Radeon RX 7800 XT": DeviceFlops(fp32=42.6*TFLOPS, fp16=85.2*TFLOPS, int8=170.4*TFLOPS),
  "AMD Radeon RX 7700 XT": DeviceFlops(fp32=34.2*TFLOPS, fp16=68.4*TFLOPS, int8=136.8*TFLOPS),
  "AMD Radeon RX 7600": DeviceFlops(fp32=21.5*TFLOPS, fp16=43.0*TFLOPS, int8=86.0*TFLOPS),
  "AMD Radeon RX 7500": DeviceFlops(fp32=16.2*TFLOPS, fp16=32.4*TFLOPS, int8=64.8*TFLOPS),
  ### Qualcomm embedded chips: TODO
  "JETSON AGX ORIN 32GB": DeviceFlops(fp32=17.65*TFLOPS, fp16=35.3*TFLOPS, int8=70.6*TFLOPS),
  "JETSON AGX ORIN 64GB": DeviceFlops(fp32=24.27*TFLOPS, fp16=48.54*TFLOPS, int8=97.09*TFLOPS),
}
CHIP_FLOPS.update({f"LAPTOP GPU {key}": value for key, value in CHIP_FLOPS.items()})
CHIP_FLOPS.update({f"Laptop GPU {key}": value for key, value in CHIP_FLOPS.items()})
CHIP_FLOPS.update({f"{key} LAPTOP GPU": value for key, value in CHIP_FLOPS.items()})
CHIP_FLOPS.update({f"{key} Laptop GPU": value for key, value in CHIP_FLOPS.items()})


async def device_capabilities() -> DeviceCapabilities:
  if psutil.MACOS:
    return await mac_device_capabilities()
  elif psutil.LINUX:
    return await linux_device_capabilities()
  elif psutil.WINDOWS:
    return await windows_device_capabilities()
  else:
    return DeviceCapabilities(
      model="Unknown Device",
      chip="Unknown Chip",
      memory=psutil.virtual_memory().total // 2**20,
      flops=DeviceFlops(fp32=0, fp16=0, int8=0),
    )

def get_jetson_device_meminfo():
  """Get memory information for Jetson devices."""
  from re import search
  from pynvml import c_nvmlMemory_t

  def extract_numeric_value(text):
    """Extract the first numeric value from a string."""
    match = search(r'\d+', text)
    return int(match.group()) if match else 0

  # Read total and free memory from /proc/meminfo
  with open("/proc/meminfo") as fp:
    total_memory = extract_numeric_value(fp.readline())
    free_memory = extract_numeric_value(fp.readline())

  # Calculate used memory
  used_memory = total_memory - free_memory

  # Return memory info object
  return c_nvmlMemory_t(
    total=total_memory * 1000,
    free=free_memory * 1000,
    used=used_memory * 1000
  )

def get_cuda_devices(device_platform="Linux") -> DeviceCapabilities:
  """
  Uses torch cuda to get available information about device. Also works with ROCm for AMD device detection and info
  """
  if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    multi_gpu_name = None

    if num_gpus > 1:
      multi_gpu_flops = DeviceFlops(fp32=0, fp16=0, int8=0)
      multi_gpu_memory_info = 0
      multi_gpu_name = f"{num_gpus} GPUs"

      for handle in range(num_gpus):
        gpu_raw_name = torch.cuda.get_device_name(handle).upper()

        if "ORIN" in gpu_raw_name and device_platform != "Windows":
          gpu_memory_info = get_jetson_device_meminfo()
          if gpu_memory_info.total > 32 * 1024 * 1024 * 1024:
            gpu_name = "JETSON AGX ORIN 64GB"
          else:
            gpu_name = "JETSON AGX ORIN 32GB"
        else:
          gpu_name = gpu_raw_name.rsplit(" ", 1)[0] if gpu_raw_name.endswith("GB") else gpu_raw_name
          gpu_memory_info = torch.cuda.get_device_properties(handle).total_memory
        
        multi_gpu_memory_info += gpu_memory_info

        gpu_flops = None
        if gpu_name in CHIP_FLOPS:
          gpu_flops = CHIP_FLOPS.get(gpu_name)
        elif "ORIN" in gpu_raw_name:
          if gpu_memory_info.total > 32 * 1024 * 1024 * 1024:
            gpu_flops = CHIP_FLOPS.get("JETSON AGX ORIN 64GB")
          else:
            gpu_flops = CHIP_FLOPS.get("JETSON AGX ORIN 32GB")

        logging.debug(f"[{gpu_name}] : {gpu_flops=}")

        if gpu_flops:
          multi_gpu_flops.fp32 += gpu_flops.fp32
          multi_gpu_flops.fp16 += gpu_flops.fp16
          multi_gpu_flops.int8 += gpu_flops.int8
        
      return DeviceCapabilities(
        model=f"{device_platform} ({multi_gpu_name})",
        chip=multi_gpu_name,
        memory=multi_gpu_memory_info // 2**20,
        flops=multi_gpu_flops,
      )

    if multi_gpu_name is None:
      handle = torch.cuda.current_device()
      gpu_raw_name = torch.cuda.get_device_name(handle).upper()
      gpu_name = gpu_raw_name.rsplit(" ", 1)[0] if gpu_raw_name.endswith("GB") else gpu_raw_name
      gpu_memory_info = torch.cuda.get_device_properties(handle).total_memory
      
      # For Jetson AGX Orin 32GB, override the GPU name
      if "ORIN" in gpu_raw_name:
        gpu_memory_info = get_jetson_device_meminfo()
        if gpu_memory_info.total > 32 * 1024 * 1024 * 1024:
          gpu_name = "JETSON AGX ORIN 64GB"
        else:
          gpu_name = "JETSON AGX ORIN 32GB"
        if DEBUG >= 1: print(f"Detected Jetson device: {gpu_raw_name} -> {gpu_name}")
      else:
        gpu_name = gpu_raw_name.rsplit(" ", 1)[0] if gpu_raw_name.endswith("GB") else gpu_raw_name
      
      # Log all keys in CHIP_FLOPS for debugging
      logging.debug(f"Available CHIP_FLOPS keys: {list(CHIP_FLOPS.keys())}")
      
      # Special handling for Jetson devices
      if "ORIN" in gpu_raw_name and device_platform != "Windows":
        logging.debug(f"Detected Jetson device: {gpu_raw_name}")
        memory_mb = gpu_memory_info.total // 1000 // 1000  # Convert to MB
        logging.debug(f"Jetson memory info: {memory_mb} MB")
      else:
        memory_mb = gpu_memory_info // 2**20

      # Always log device info for debugging
      logging.debug(f"NVIDIA device {gpu_name=} {memory_mb=}")
      
      # Get FLOPS values and log for debugging
      flops = CHIP_FLOPS.get(gpu_name, DeviceFlops(fp32=0, fp16=0, int8=0))
      logging.debug(f"FLOPS for {gpu_name}: {flops}")
      
      # Check if the key exists in CHIP_FLOPS
      if gpu_name in CHIP_FLOPS:
        logging.debug(f"Found {gpu_name} in CHIP_FLOPS")
      else:
        logging.debug(f"WARNING: {gpu_name} not found in CHIP_FLOPS")
        # Try to find similar keys
        similar_keys = [k for k in CHIP_FLOPS.keys() if "ORIN" in k or "JETSON" in k]
        logging.debug(f"Similar keys in CHIP_FLOPS: {similar_keys}")
        
        # Try a direct assignment for Jetson
        if "ORIN" in gpu_raw_name:
          gpu_memory_info = get_jetson_device_meminfo()
          if gpu_memory_info.total > 32 * 1024 * 1024 * 1024:
            logging.debug("Forcing JETSON AGX ORIN 64GB FLOPS values")
            flops = CHIP_FLOPS.get("JETSON AGX ORIN 64GB")
          else:
            logging.debug("Forcing JETSON AGX ORIN 32GB FLOPS values")
            flops = CHIP_FLOPS.get("JETSON AGX ORIN 32GB")

      # Log the final values being returned
      logging.debug(f"Returning DeviceCapabilities with model={gpu_name}, memory={memory_mb}, flops={flops}")
    
      return DeviceCapabilities(
        model=f"{device_platform} ({gpu_name})",
        chip=gpu_name,
        memory=memory_mb,
        flops=flops,
      )

  # Fallback for CPU
  return DeviceCapabilities(
    model=f"{device_platform}",
    chip=f"CPU ({platform.processor()})",
    memory=psutil.virtual_memory().total // 2**20,
    flops=DeviceFlops(fp32=0, fp16=0, int8=0),
  )

def get_amd_devices(device_platform="Linux") -> DeviceCapabilities:
  """
  Detection of AMD GPU info using pyamdgpuinfo if installed
  """
  import pyamdgpuinfo
  has_amd = True
  
  gpu_raw_info = pyamdgpuinfo.get_gpu(0)
  gpu_name = gpu_raw_info.name
  gpu_memory_info = gpu_raw_info.memory_info["vram_size"]

  if DEBUG >= 2: logging.debug(f"AMD device {gpu_name=} {gpu_memory_info=}")

  return DeviceCapabilities(
    model=f"{device_platform} ({gpu_name})",
    chip=gpu_name,
    memory=gpu_memory_info // 2**20,
    flops=CHIP_FLOPS.get(gpu_name, DeviceFlops(fp32=0, fp16=0, int8=0)),
  )

async def mac_device_capabilities() -> DeviceCapabilities:
  model_id, chip_id, memory = await get_mac_system_info()
  
  return DeviceCapabilities(
    model=model_id,
    chip=chip_id,
    memory=memory,
    flops=CHIP_FLOPS.get(chip_id, DeviceFlops(fp32=0, fp16=0, int8=0))
  )

async def linux_device_capabilities() -> DeviceCapabilities:
  if torch.cuda.is_available():
    return get_cuda_devices()
  
  # try AMD
  try:
    return get_amd_devices()
  except (ImportError, Exception) as e:
    if DEBUG >= 2: logging.debug(f"AMD GPU detection failed:\n{e}")

    
  # Fallback for CPU
  return DeviceCapabilities(
    model=f"Linux",
    chip=f"CPU ({platform.processor()})",
    memory=psutil.virtual_memory().total // 2**20,
    flops=DeviceFlops(fp32=0, fp16=0, int8=0),
  )

async def windows_device_capabilities() -> DeviceCapabilities:
  import win32com.client  # install pywin32
  
  if torch.cuda.is_available():
    return get_cuda_devices(device_platform="Windows")
  
  # try AMD
  try:
    return get_amd_devices(device_platform="Windows")
  except (ImportError, Exception) as e:
    if DEBUG >= 2: logging.debug(f"AMD GPU detection failed:\n{e}")
  
  return DeviceCapabilities(
    model=f"Windows",
    chip=f"CPU ({platform.processor()})",
    memory=psutil.virtual_memory().total // 2**20,
    flops=DeviceFlops(fp32=0, fp16=0, int8=0),
  )




