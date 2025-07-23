import sys
import subprocess
import re

from setuptools import find_packages, setup

# Base requirements for all platforms
install_requires = [
  "aiohttp>=3.10.11",
  "aiohttp_cors>=0.7.0",
  "aiofiles>=24.1.0",
  "grpcio>=1.71.0",
  "grpcio-tools>=1.71.0",
  "Jinja2>=3.1.4",
  "numpy>=2.0.0",
  "nuitka>=2.5.1",
  "opencv-python>=4.10.0.84",
  "pillow>=10.4.0",
  "prometheus-client>=0.20.0",
  "protobuf>=5.28.1",
  "psutil>=6.0.0",
  "pydantic>=2.9.2",
  "requests>=2.32.3",
  "rich>=13.7.1",
  "scapy>=2.6.1",
  "tqdm>=4.66.4",
  "transformers>=4.46.3",
  "uuid>=1.30",
  "accelerate>=0.34.2",
  "pytest>=8.3.3",
  "pytest-asyncio>=0.24.0",
  "scapy>=2.6.1",
]

extras_require = {
  "formatting": ["yapf==0.40.2",],
  "windows": ["pywin32==308","winloop==0.1.8"],
  "nvidia-gpu": ["nvidia-ml-py==12.560.30",],
  "amd-gpu": ["pyrsmi==0.2.0"],
  "non-windows": ["uvloop==0.21.0"],
  "jetson": ["torchtune>=0.6.1", "torchao>=0.11.0"],
  "torch-rocm63": [
    "torch @ https://download.pytorch.org/whl/rocm6.3",
    "torchvision @ https://download.pytorch.org/whl/rocm6.3",
    "torchaudio @ https://download.pytorch.org/whl/rocm6.3",
    "torchtune>=0.6.1",
    "torchao>=0.11.0"
  ],
  "torch-cuda128": [
    "torch @ https://download.pytorch.org/whl/cu128",
    "torchvision @ https://download.pytorch.org/whl/cu128",
    "torchaudio @ https://download.pytorch.org/whl/cu128",
    "torchtune>=0.6.1",
    "torchao>=0.11.0"
  ],
  "torch-cuda118": [
    "torch @ https://download.pytorch.org/whl/cu118",
    "torchvision @ https://download.pytorch.org/whl/cu118",
    "torchaudio @ https://download.pytorch.org/whl/cu118",
    "torchtune>=0.6.1",
    "torchao>=0.11.0"
  ],
  "torch-default": [
    "torch", 
    "torchvision", 
    "torchaudio",
    "torchtune>=0.6.1",
    "torchao>=0.11.0"
  ]
}

use_win = False
# Check if running Windows
if sys.platform.startswith("win32"):
  install_requires.extend(extras_require["windows"])
  use_win = True

if not use_win:
  install_requires.extend(extras_require["non-windows"])



def _add_gpu_requires():
  global install_requires
  # Add Nvidia-GPU
  try:
    out = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], shell=True, text=True, capture_output=True, check=False)
    if out.returncode == 0:
      install_requires.extend(extras_require["nvidia-gpu"])
      
      # Check if running on Jetson platform
      is_jetson = False
      try:
        # Check for Jetson-specific file
        import os
        if os.path.exists('/etc/nv_tegra_release'):
          is_jetson = True
        # Alternative check for Jetson hardware
        elif out.stdout and any(jetson_keyword in out.stdout.lower() for jetson_keyword in ['tegra', 'jetson']):
          is_jetson = True
          
        if is_jetson:
          print("Detected Jetson platform, adding Jetson-specific dependencies")
          install_requires.extend(extras_require["jetson"])
      except Exception as e:
        print(f"Error detecting Jetson platform: {e}")
  except subprocess.CalledProcessError:
    pass

  # Add AMD-GPU
  # This will mostly work only on Linux, amd/rocm-smi is not yet supported on Windows
  try:
    out = subprocess.run(['amd-smi', 'list', '--csv'], shell=True, text=True, capture_output=True, check=False)
    if out.returncode == 0:
      install_requires.extend(extras_require["amd-gpu"])
  except:
    out = subprocess.run(['rocm-smi', 'list', '--csv'], shell=True, text=True, capture_output=True, check=False)
    if out.returncode == 0:
      install_requires.extend(extras_require["amd-gpu"])
  finally:
    pass

def _pytorch_install():
  """
  Select the correct pytorch install.
  Will need to be updated if new versions of pytorch, cuda or rocm
  """
  try:
    out = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], shell=True, text=True, capture_output=True, check=False)
    if out.returncode == 0:
      cuda_result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True)
      cuda_out = cuda_result.stdout
      match_ver = re.search(r'release (\d+\.\d+)', cuda_out)
      if match_ver:
        if match_ver == "12.8":
          install_requires.extend(extras_require["torch-cuda128"])
        elif match_ver == "11.8":
          install_requires.extend(extras_require["torch-cuda118"])
        else:
          install_requires.extend(extras_require["torch-default"])
      else:
        install_requires.extend(extras_require["torch-default"])
    else:
      out = subprocess.run(['amd-smi', 'list', '--csv'], shell=True, text=True, capture_output=True, check=False)
      if out.returncode == 0:
        install_requires.extend(extras_require["torch-rocm63"])
      else:
        install_requires.extend(extras_require["torch-default"])
  except Exception as err:
    install_requires.extend(extras_require["torch-default"])

_pytorch_install()
_add_gpu_requires()

setup(
  name="xotorch",
  version="1.1.0",
  packages=find_packages(),
  install_requires=install_requires,
  extras_require=extras_require,
  package_data={"xotorch": ["tinychat/**/*"]},
  entry_points={"console_scripts": ["xot = xotorch.main:run"]},
)
