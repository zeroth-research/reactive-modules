"""Live visualisation demo.

Opens a browser at http://localhost:7777 and steps through a SimpleEnv driven
by a SimpleQNet, showing the reactive module graph with live wire-value overlays.

Run with:
    cd python && uv run python scripts/ui/ui_visual_demo.py
"""
import sys
sys.path.insert(0, "tests")

import time
import torch
import zrth.visual

from gym.environments import SimpleEnv
from gym.qnetworks import SimpleQNet

# ── 1. Start the server and open the browser ─────────────────────────────────
zrth.visual.start(port=7777)
print("Browser opened at http://localhost:7777")
print("Keep this window open — the graph will update as you run the script.\n")

time.sleep(0.5)   # let the browser load

# ── 2. Instantiate the env — graph is pushed automatically ───────────────────
print("Defining SimpleEnv …")
env = SimpleEnv()
obs, reward, terminated, truncated = env.reset()
print(f"  reset → obs={obs}  reward={reward}\n")

time.sleep(1.0)

# ── 3. Define and push a Q-network ───────────────────────────────────────────
print("Defining SimpleQNet …")
qnet = SimpleQNet(state_size=1, action_size=2, hidden_size=2)
time.sleep(1.0)

# ── 4. Step the env, pushing live wire values each time ──────────────────────
pulse    = torch.tensor([0.0, 1.0])   # action 1 → move right
no_pulse = torch.tensor([1.0, 0.0])  # action 0 → move left

print("Stepping env (watch the browser for live wire-value overlays) …")
for step in range(6):
    q_values = qnet.forward(torch.tensor([float(obs)]))
    obs, reward, terminated, truncated = env.step(q_values)
    print(f"  step {step+1}: obs={obs}  reward={reward}  done={terminated}")
    time.sleep(1.2)
    if terminated:
        print("  Goal reached!")
        break

print("\nDone.  The browser tab stays open — refresh to reconnect.")
