"""
Work in progress viz change, changing from using rich
"""
import math
from collections import OrderedDict
from typing import List, Optional, Tuple, Dict
from xotorch.helpers import xotorch_text, pretty_print_bytes, pretty_print_bytes_per_second
from xotorch.topology.topology import Topology
from xotorch.topology.partitioning_strategy import Partition
from xotorch.download.download_progress import RepoProgressEvent
from xotorch.topology.device_capabilities import UNKNOWN_DEVICE_CAPABILITIES
from rich.console import Console, Group
from rich.text import Text
from rich.live import Live
from rich.style import Style
from rich.table import Table
from rich.layout import Layout
from rich.syntax import Syntax
from rich.panel import Panel
from rich.markdown import Markdown
from rich.columns import Columns

class TopologyViz:
  def __init__(self, chatgpt_api_endpoints: List[str] = [], web_chat_urls: List[str] = []):
    self.chatgpt_api_endpoints = chatgpt_api_endpoints
    self.web_chat_urls = web_chat_urls
    self.topology = Topology()
    self.partitions: List[Partition] = []
    self.node_id = None
    self.node_download_progress: Dict[str, RepoProgressEvent] = {}
    self.requests: OrderedDict[str, Tuple[str, str]] = {}

    self.console = Console()
    self.console_width = self.console.size.width
    self.console_height = self.console.size.height
    self.layout = Layout()
    self.layout.split(Layout(name="main"), Layout(name="prompt_output", size=15), Layout(name="download", size=25))
    
    self.ninfo_panel = Panel(self._generate_ninfo_layout(), title="", border_style="red1")
    self.ninfo_panel.width = int(self.console_width/2) - 15
    self.ninfo_panel.height = 15

    self.node_panel = Panel(self._generate_node_panel(), title="0 Nodes")
    self.prompt_output_panel = Panel("", title="Prompt and Output", border_style="deep_pink4")
    self.download_panel = Panel("", title="Download Progress", border_style="bright_white")
    
    self.layout["main"].update(
      Columns([
        self.ninfo_panel,
        self.node_panel
      ])
    )

    self.layout["prompt_output"].update(self.prompt_output_panel)
    self.layout["download"].update(self.download_panel)
    

    # Initially hide the prompt_output panel
    self.layout["prompt_output"].visible = False
    self.live_panel = Live(self.layout, auto_refresh=False, console=self.console)
    self.live_panel.start()
  def update_visualization(self, topology: Topology, partitions: List[Partition], node_id: Optional[str] = None, node_download_progress: Dict[str, RepoProgressEvent] = {}):
    self.topology = topology
    self.partitions = partitions
    self.node_id = node_id
    if node_download_progress:
      self.node_download_progress = node_download_progress
    self.refresh()

  def update_prompt(self, request_id: str, prompt: Optional[str] = None):
    self.requests[request_id] = [prompt, self.requests.get(request_id, ["", ""])[1]]
    self.refresh()

  def update_prompt_output(self, request_id: str, output: Optional[str] = None):
    self.requests[request_id] = [self.requests.get(request_id, ["", ""])[0], output]
    self.refresh()

  def refresh(self):
    self.node_panel = self._generate_node_panel()
    self.node_panel.width = int(self.console_width/2) + 12

    node_amt = len(self.topology.nodes)
    self.node_panel.title = f"1 Node" if node_amt == 1 else f"{node_amt} Nodes"

    self.layout["main"].update(
      Columns([
        self.ninfo_panel,
        self.node_panel
      ])
    )

    # Update and show/hide prompt and output panel
    if any(r[0] or r[1] for r in self.requests.values()):
      self.prompt_output_panel = self._generate_prompt_output_layout()
      self.layout["prompt_output"].update(self.prompt_output_panel)
      self.layout["prompt_output"].visible = True
    else:
      self.layout["prompt_output"].visible = False

    # Only show download_panel if there are in-progress downloads
    if any(progress.status == "in_progress" for progress in self.node_download_progress.values()):
      self.download_panel.renderable = self._generate_download_layout()
      self.layout["download"].visible = True
    else:
      self.layout["download"].visible = False

    self.live_panel.update(self.layout, refresh=True)

  def _generate_prompt_output_layout(self) -> Panel:
    content = []
    requests = list(self.requests.values())[-3:]  # Get the 3 most recent requests
    max_width = int(self.console_width/2)  # Full width minus padding and icon

    # Calculate available height for content
    panel_height = self.console_height  # Fixed panel height
    available_lines = panel_height - 2  # Subtract 2 for panel borders
    lines_per_request = available_lines // len(requests) if requests else 0

    for (prompt, output) in reversed(requests):
      # prompt_icon, output_icon = "💬️", "🤖"

      # Equal space allocation for prompt and output
      max_prompt_lines = lines_per_request // 2
      max_output_lines = lines_per_request - max_prompt_lines - 1  # -1 for spacing

      # Process prompt
      prompt_lines = []
      for line in prompt.split('\n'):
        words = line.split()
        current_line = []
        current_length = 0

        for word in words:
          if current_length + len(word) + 1 <= max_width:
            current_line.append(word)
            current_length += len(word) + 1
          else:
            if current_line:
              prompt_lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)

        if current_line:
          prompt_lines.append(' '.join(current_line))

      # Truncate prompt if needed
      if len(prompt_lines) > max_prompt_lines:
        prompt_lines = prompt_lines[:max_prompt_lines]
        if prompt_lines:
          last_line = prompt_lines[-1]
          if len(last_line) + 4 <= max_width:
            prompt_lines[-1] = last_line + " ..."
          else:
            prompt_lines[-1] = last_line[:max_width-4] + " ..."

      prompt_lines = '\n'.join(prompt_lines)
      prompt_text = Text(f"[Prompt]\n", style="bold grey70")
      prompt_text.append(f"{prompt_lines}", style="deep_sky_blue1")
      content.append(prompt_text)

      # Process output with similar word wrapping
      if output:  # Only process output if it exists
        output_lines = []
        for line in output.split('\n'):
          words = line.split()
          current_line = []
          current_length = 0

          for word in words:
            if current_length + len(word) + 1 <= max_width:
              current_line.append(word)
              current_length += len(word) + 1
            else:
              if current_line:
                output_lines.append(' '.join(current_line))
              current_line = [word]
              current_length = len(word)

          if current_line:
            output_lines.append(' '.join(current_line))

        # Truncate output if needed
        if len(output_lines) > max_output_lines:
          output_lines = output_lines[:max_output_lines]
          if output_lines:
            last_line = output_lines[-1]
            if len(last_line) + 4 <= max_width:
              output_lines[-1] = last_line + " ..."
            else:
              output_lines[-1] = last_line[:max_width-4] + " ..."

        output_lines = '\n'.join(output_lines)
        output_text = Text(f"[Response]\n", style="bold grey70")
        output_text.append(f"{output_lines}\n", style="white")
        content.append(output_text)

      content.append(Text())  # Empty line between entries

    return Panel(
      Group(*content),
      title="Chat",
      border_style="orange1"
    )
  
  def _generate_node_panel(self) -> Panel:
    node_info = []

    for _, partition in enumerate(self.partitions):
      device_capabilities = self.topology.nodes.get(partition.node_id, UNKNOWN_DEVICE_CAPABILITIES)

      # Place node with different color for active node and this node
      if partition.node_id == self.topology.active_node_id:
        active_status = "🟢"
      else:
        active_status = "🟡"

      # Place node info (model, memory, TFLOPS, partition) on three lines
      node_text = Text(f"""
        [{active_status}] {device_capabilities.model} {device_capabilities.memory // 1024}GB    
        {device_capabilities.flops.fp16}TFLOPS
        [{partition.start:.2f}-{partition.end:.2f}]
      """)

      node_info.append(node_text)
      node_info.append(Text())

    return Panel(
      Group(*node_info),
      title="Nodes",
      border_style="white"
    )

  def _generate_ninfo_layout(self) -> str:
    # Calculate visualization parameters
    num_partitions = len(self.partitions)
    radius_x = 30
    radius_y = 12
    center_x, center_y = 20, 24  # Increased center_y to add more space

    # Generate visualization
    visualization = [[" " for _ in range(100)] for _ in range(48)]  # Increased height to 48

    # Add xotorch_text at the top in red
    xotorch_lines = xotorch_text.split("\n")
    red_style = Style(color="red1")
    max_line_length = max(len(line) for line in xotorch_lines)
    for i, line in enumerate(xotorch_lines):
      centered_line = line.center(max_line_length)
      start_x = 10 
      colored_line = Text(centered_line, style=red_style)
      for j, char in enumerate(str(colored_line)):
        if 0 <= j < 100 and i < len(visualization):
          visualization[i][start_x + j] = char

    # Display chatgpt_api_endpoints and web_chat_urls
    info_lines = []
    if len(self.web_chat_urls) > 0:
      info_lines.append(f"Web Chat URL (tinychat): {' '.join(self.web_chat_urls[:1])}")
    if len(self.chatgpt_api_endpoints) > 0:
      info_lines.append(f"ChatGPT API endpoint: {' '.join(self.chatgpt_api_endpoints[:1])}")

    info_start_y = len(xotorch_lines) + 1
    info_start_x = 2
    for i, line in enumerate(info_lines):
      for j, char in enumerate(line):
        if 0 <= j < 100 and info_start_y + i < 48:
          visualization[info_start_y + i][info_start_x + j] = char

    # # Calculate total FLOPS and position on the bar
    # total_flops = sum(self.topology.nodes.get(partition.node_id, UNKNOWN_DEVICE_CAPABILITIES).flops.fp16 for partition in self.partitions)
    # bar_pos = (math.tanh(total_flops**(1/3)/2.5 - 2) + 1)

    # # Add GPU poor/rich bar
    # bar_width = 30
    # bar_start_x = (100-bar_width) // 2
    # bar_y = info_start_y + len(info_lines) + 1

    # # Create a gradient bar using emojis
    # gradient_bar = Text()
    # emojis = ["🟥", "🟧", "🟨", "🟩"]
    # for i in range(bar_width):
    #   emoji_index = min(int(i/(bar_width/len(emojis))), len(emojis) - 1)
    #   gradient_bar.append(emojis[emoji_index])

    # # Add the gradient bar to the visualization
    # visualization[bar_y][bar_start_x - 1] = "["
    # visualization[bar_y][bar_start_x + bar_width] = "]"
    # for i, segment in enumerate(str(gradient_bar)):
    #   visualization[bar_y][bar_start_x + i] = segment

    # # Add labels
    # visualization[bar_y - 1][bar_start_x - 10:bar_start_x - 3] = "GPU poor"
    # visualization[bar_y - 1][bar_start_x + bar_width*2 + 2:bar_start_x + bar_width*2 + 11] = "GPU rich"

    # # Add position indicator and FLOPS value
    # pos_x = bar_start_x + int(bar_pos*bar_width)
    # flops_str = f"{total_flops:.2f} TFLOPS"
    # visualization[bar_y - 1][pos_x] = "▼"
    # visualization[bar_y + 1][pos_x - len(flops_str) // 2:pos_x + len(flops_str) // 2 + len(flops_str) % 2] = flops_str
    # visualization[bar_y + 2][pos_x] = "▲"

    # # Add an extra empty line for spacing
    # bar_y += 4

    # Convert to string
    return "\n".join("".join(str(char) for char in row) for row in visualization)

  def _generate_download_layout(self) -> Table:
    summary = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    summary.add_column("Info", style="cyan", no_wrap=True, ratio=50)
    summary.add_column("Progress", style="cyan", no_wrap=True, ratio=40)
    summary.add_column("Percentage", style="cyan", no_wrap=True, ratio=10)

    # Current node download progress
    if self.node_id in self.node_download_progress:
      download_progress = self.node_download_progress[self.node_id]
      title = f"Downloading model {download_progress.repo_id}@{download_progress.repo_revision} ({download_progress.completed_files}/{download_progress.total_files}):"
      summary.add_row(Text(title, style="bold"))
      progress_info = f"{pretty_print_bytes(download_progress.downloaded_bytes)} / {pretty_print_bytes(download_progress.total_bytes)} ({pretty_print_bytes_per_second(download_progress.overall_speed)})"
      summary.add_row(progress_info)

      eta_info = f"{download_progress.overall_eta}"
      summary.add_row(eta_info)

      summary.add_row("")  # Empty row for spacing

      for file_path, file_progress in download_progress.file_progress.items():
        if file_progress.status != "complete":
          progress = int(file_progress.downloaded/file_progress.total*30)
          bar = f"[{'=' * progress}{' ' * (30 - progress)}]"
          percentage = f"{file_progress.downloaded / file_progress.total * 100:.0f}%"
          summary.add_row(Text(file_path[:30], style="cyan"), bar, percentage)

    summary.add_row("")  # Empty row for spacing

    # Other nodes download progress summary
    summary.add_row(Text("Other Nodes Download Progress:", style="bold"))
    for node_id, progress in self.node_download_progress.items():
      if node_id != self.node_id:
        device = self.topology.nodes.get(node_id)
        partition = next((p for p in self.partitions if p.node_id == node_id), None)
        partition_info = f"[{partition.start:.2f}-{partition.end:.2f}]" if partition else ""
        percentage = progress.downloaded_bytes/progress.total_bytes*100 if progress.total_bytes > 0 else 0
        speed = pretty_print_bytes_per_second(progress.overall_speed)
        device_info = f"{device.model if device else 'Unknown Device'} {device.memory // 1024 if device else '?'}GB {partition_info}"
        progress_info = f"{progress.repo_id}@{progress.repo_revision} ({speed})"
        progress_bar = f"[{'=' * int(percentage // 3.33)}{' ' * (30 - int(percentage // 3.33))}]"
        percentage_str = f"{percentage:.1f}%"
        eta_str = f"{progress.overall_eta}"
        summary.add_row(device_info, progress_info, percentage_str)
        summary.add_row("", progress_bar, eta_str)

    return summary
