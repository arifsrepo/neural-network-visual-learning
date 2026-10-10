"""
Step 8: Animated Neuron Firing Simulation

This script visually animates the full lifecycle of a neuron firing:
1. Signal Propagation: Input signals travel along synapses with specific weights.
2. Accumulation: The soma (neuron cell body) aggregates weighted inputs + bias (z = w·x + b).
3. Action Potential / Firing: If z >= 0, the neuron flashes and fires an output spike!
   If z < 0, the neuron remains dormant/inhibited.
4. It cycles through all 4 inputs of an AND gate ([0,0], [0,1], [1,0], [1,1]) so you can watch
   when the threshold is reached.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Configure AND gate dataset and neuron weights
samples = [
    {"x": [0, 0], "label": "[0, 0]"},
    {"x": [0, 1], "label": "[0, 1]"},
    {"x": [1, 0], "label": "[1, 0]"},
    {"x": [1, 1], "label": "[1, 1]"}
]

weights = np.array([0.5, 0.5])
bias = -0.7

# Geometry coordinates
pos_x1 = np.array([0.15, 0.70])
pos_x2 = np.array([0.15, 0.30])
pos_soma = np.array([0.52, 0.50])
pos_out = np.array([0.88, 0.50])

# Animation settings
FRAMES_PER_SAMPLE = 40
TOTAL_FRAMES = FRAMES_PER_SAMPLE * len(samples)

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.canvas.manager.set_window_title("Neuron Firing Animation")

def init_plot():
    ax.clear()
    ax.set_facecolor("#1e1e2e")
    fig.patch.set_facecolor("#181825")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

# Pre-draw static layout elements
def draw_base_network():
    # Synapse 1 line
    ax.plot([pos_x1[0], pos_soma[0]], [pos_x1[1], pos_soma[1]], 
            color="#45475a", lw=3, zorder=1)
    # Synapse 2 line
    ax.plot([pos_x2[0], pos_soma[0]], [pos_x2[1], pos_soma[1]], 
            color="#45475a", lw=3, zorder=1)
    # Axon line (output)
    ax.plot([pos_soma[0], pos_out[0]], [pos_soma[1], pos_out[1]], 
            color="#45475a", lw=3, zorder=1)

    # Synapse weight labels
    ax.text(0.31, 0.64, f"w1 = {weights[0]:.1f}", color="#89b4fa", fontsize=11, fontweight="bold")
    ax.text(0.31, 0.36, f"w2 = {weights[1]:.1f}", color="#89b4fa", fontsize=11, fontweight="bold")
    ax.text(0.52, 0.37, f"bias = {bias:.1f}", color="#fab387", fontsize=11, ha="center")

    # Titles
    ax.text(0.50, 0.94, "Neuron Firing & Action Potential Simulation", color="#cdd6f4",
            fontsize=15, fontweight="bold", ha="center")
    ax.text(0.50, 0.88, "Task: AND Gate Logic  |  Activation: Threshold (Step)", color="#a6adc8",
            fontsize=11, ha="center")

def update(frame):
    ax.clear()
    ax.set_facecolor("#1e1e2e")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    draw_base_network()

    sample_idx = (frame // FRAMES_PER_SAMPLE) % len(samples)
    sub_frame = frame % FRAMES_PER_SAMPLE

    sample = samples[sample_idx]
    x = sample["x"]
    z = x[0] * weights[0] + x[1] * weights[1] + bias
    fires = (z >= 0)

    # 1. Input Nodes
    c1 = "#a6e3a1" if x[0] == 1 else "#585b70"
    c2 = "#a6e3a1" if x[1] == 1 else "#585b70"
    
    # Draw input circles
    ax.scatter([pos_x1[0]], [pos_x1[1]], s=700, color=c1, edgecolors="#cdd6f4", lw=2, zorder=3)
    ax.scatter([pos_x2[0]], [pos_x2[1]], s=700, color=c2, edgecolors="#cdd6f4", lw=2, zorder=3)
    ax.text(pos_x1[0], pos_x1[1], f"x1={x[0]}", color="#11111b" if x[0] else "#cdd6f4",
            ha="center", va="center", fontweight="bold", fontsize=10, zorder=4)
    ax.text(pos_x2[0], pos_x2[1], f"x2={x[1]}", color="#11111b" if x[1] else "#cdd6f4",
            ha="center", va="center", fontweight="bold", fontsize=10, zorder=4)

    # 2. Phase: Synaptic Pulses (Frames 0 to 18)
    if sub_frame <= 18:
        progress = sub_frame / 18.0
        # Pulse along synapse 1
        if x[0] > 0:
            p1 = pos_x1 + (pos_soma - pos_x1) * progress
            ax.scatter([p1[0]], [p1[1]], s=160, color="#f9e2af", edgecolors="#fab387", lw=1.5, zorder=5)
        # Pulse along synapse 2
        if x[1] > 0:
            p2 = pos_x2 + (pos_soma - pos_x2) * progress
            ax.scatter([p2[0]], [p2[1]], s=160, color="#f9e2af", edgecolors="#fab387", lw=1.5, zorder=5)

    # 3. Phase: Soma Accumulation & Firing (Frames 18 to 40)
    soma_color = "#313244"
    soma_edge = "#cdd6f4"
    soma_size = 1400

    if sub_frame > 18:
        if fires:
            # Pulsing / Glowing when fired!
            glow_intensity = 0.5 + 0.5 * np.sin((sub_frame - 18) * 0.4)
            soma_color = "#f9e2af"  # Bright electric yellow
            soma_edge = "#f38ba8"
            soma_size = 1400 + int(250 * glow_intensity)
            # Action potential shockwave ring
            ring_radius = 0.05 + 0.06 * ((sub_frame - 18) / 22.0)
            circle = plt.Circle(pos_soma, ring_radius, color="#f9e2af", fill=False,
                                lw=2.5, alpha=max(0.0, 1.0 - (sub_frame - 18)/22.0), zorder=2)
            ax.add_patch(circle)
        else:
            # Dormant / Inhibited state (dim red-tinted)
            soma_color = "#45475a"
            soma_edge = "#f38ba8"

    ax.scatter([pos_soma[0]], [pos_soma[1]], s=soma_size, color=soma_color,
               edgecolors=soma_edge, lw=2.5, zorder=3)
    ax.text(pos_soma[0], pos_soma[1] + 0.015, "Σ + Act",
            color="#11111b" if fires and sub_frame > 18 else "#cdd6f4",
            ha="center", va="center", fontweight="bold", fontsize=11, zorder=4)
    ax.text(pos_soma[0], pos_soma[1] - 0.025, f"z={z:+.2f}",
            color="#11111b" if fires and sub_frame > 18 else "#fab387",
            ha="center", va="center", fontweight="bold", fontsize=9, zorder=4)

    # 4. Phase: Axon Signal & Output Node (Frames 22 to 40)
    out_color = "#313244"
    out_text_color = "#6c7086"
    out_val = 0

    if sub_frame > 22 and fires:
        axon_progress = min(1.0, (sub_frame - 22) / 14.0)
        p_axon = pos_soma + (pos_out - pos_soma) * axon_progress
        ax.scatter([p_axon[0]], [p_axon[1]], s=200, color="#a6e3a1", edgecolors="#94e2d5", lw=2, zorder=5)

        if axon_progress >= 0.85:
            out_color = "#a6e3a1"
            out_text_color = "#11111b"
            out_val = 1
    elif sub_frame > 22 and not fires:
        out_color = "#313244"
        out_text_color = "#f38ba8"
        out_val = 0

    # Output node circle
    ax.scatter([pos_out[0]], [pos_out[1]], s=900, color=out_color,
               edgecolors="#cdd6f4", lw=2, zorder=3)
    ax.text(pos_out[0], pos_out[1], f"Out={out_val}", color=out_text_color,
            ha="center", va="center", fontweight="bold", fontsize=11, zorder=4)

    # 5. Status Banner at Bottom
    if sub_frame <= 18:
        status_text = f"Sample {sample['label']}: Incoming signals propagating along synapses..."
        status_color = "#89b4fa"
    elif fires:
        status_text = f"Sample {sample['label']}: THRESHOLD REACHED (z = {z:+.2f} >= 0) -> NEURON FIRES! 🔥"
        status_color = "#a6e3a1"
    else:
        status_text = f"Sample {sample['label']}: SUB-THRESHOLD (z = {z:+.2f} < 0) -> INHIBITED / DORMANT ❄️"
        status_color = "#eba0ac"

    ax.text(0.50, 0.08, status_text, color=status_color, fontsize=12,
            fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#181825", edgecolor=status_color, lw=1.5))

anim = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=50, repeat=True)

print("=== Neuron Firing Visualizer ===")
print("Watching a neuron process each input pattern of an AND gate:")
for s in samples:
    calc_z = s['x'][0]*weights[0] + s['x'][1]*weights[1] + bias
    fired = calc_z >= 0
    print(f"  Input {s['label']}: z = {calc_z:+.2f} -> {'FIRES (1)' if fired else 'DORMANT (0)'}")
print("\nLaunching animated window...")
plt.show()
