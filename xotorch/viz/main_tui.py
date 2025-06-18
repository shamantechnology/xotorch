"""
Main TUI for xotorch
Displays tinychat and API URLs along with node and token throughput information
Will be an interface to the Chat TUI
"""
from typing import List

from textual.app import App, ComposeResult, RenderResult
from textual.widgets import Footer, Static, Button, Footer, Label, Header
from textual.screen import Screen
from textual.containers import Grid

from xotorch.orchestration.node import Node
from xotorch.topology.device_capabilities import get_cuda_devices
from xotorch.topology.topology import Topology
    
class XOTFooter(Footer):
  def render(self) -> str:
      return "[b]XOTORCH v1[/b]"

class InfoPopup(Screen):
  """Screen with info about Chat and API URLs along with verison info"""
  BINDINGS = [("escape", "app.pop_screen", "Pop screen")]

  def compose(self) -> ComposeResult:
    yield Static("Chat: https://localhost\nAPI: https://localhost", id="infopopup")

  def key_escape(self) -> None:
    self.app.pop_screen()  # Close on Escape key

class QuitScreen(Screen):
  """Screen with a dialog to quit."""

  def compose(self) -> ComposeResult:
    yield Grid(
      Label("Are you sure you want to quit?", id="question"),
      Button("Quit", variant="error", id="quit"),
      Button("Cancel", variant="primary", id="cancel"),
      id="dialog",
    )

  def on_button_pressed(self, event: Button.Pressed) -> None:
    if event.button.id == "quit":
      self.app.exit()
    else:
      self.app.pop_screen()

class CurrentDeviceInfo(Static):
  """Info about current device"""

  def render(self) -> RenderResult:
    dc = get_cuda_devices()
    model_info = f"Model: {dc.model}\nChip: {dc.chip}\nMem (MB): {dc.memory}\nFlops: {str(dc.flops)}"

    return model_info

class XOTApp(App):
  CSS_PATH = "main.tcss"
  SCREENS = {"infopopup": InfoPopup, "quitscreen": QuitScreen}

  BINDINGS = [
    ("q", "push_screen('quitscreen')", "Quit"),
    ("ctrl+q", "push_screen('quitscreen')", "Quit"),
    ("i", "push_screen('infopopup')", "Info"),  # Press 'i' to show popup
  ]

  def __init__(
    self,
    nodes: Node = None,
    chatgpt_api_endpoints: List[str] = [],
    web_chat_urls: List[str] = []
  ):
    super().__init__()

    self.nodes = nodes
    self.chatgpt_api_endpoints = chatgpt_api_endpoints
    self.web_chat_urls = web_chat_urls
    self.topology = Topology()

    print(self.topology.nodes)

  def compose(self) -> ComposeResult:
    yield Header(show_clock=True)
    yield CurrentDeviceInfo(id="cdinfo")
    yield XOTFooter()

  def on_mount(self) -> None:
    self.title = "XOTORCH"

