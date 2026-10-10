import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

samples = [
    {"x": [0, 0], "label": "[0, 0]"},
    {"x": [0, 1], "label": "[0, 1]"},
    {"x": [1, 0], "label": "[1, 0]"},
    {"x": [1, 1], "label": "[1, 1]"}
]
weights = np.array([0.5, 0.5])
bias = -0.7

pos_x1 = np.array([0.15, 0.70])
pos_x2 = np.array([0.15, 0.30])
pos_soma = np.array([0.52, 0.50])
pos_out = np.array([0.88, 0.50])

frames = []
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=100)

for s in samples:
    x = s["x"]
    z = x[0]*weights[0] + x[1]*weights[1] + bias
    fires = (z >= 0)
    
    for t_step in range(14):
        ax.clear()
        ax.set_facecolor("#111827")
        fig.patch.set_facecolor("#0b0f19")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        
        # Connections
        ax.plot([pos_x1[0], pos_soma[0]], [pos_x1[1], pos_soma[1]], color="#334155", lw=3)
        ax.plot([pos_x2[0], pos_soma[0]], [pos_x2[1], pos_soma[1]], color="#334155", lw=3)
        ax.plot([pos_soma[0], pos_out[0]], [pos_soma[1], pos_out[1]], color="#334155", lw=3)
        
        # Labels
        ax.text(0.31, 0.65, f"w1 = {weights[0]}", color="#38bdf8", fontsize=10, fontweight="bold")
        ax.text(0.31, 0.35, f"w2 = {weights[1]}", color="#38bdf8", fontsize=10, fontweight="bold")
        ax.text(0.52, 0.36, f"bias = {bias}", color="#fb923c", fontsize=10, ha="center")
        ax.text(0.50, 0.93, "Neuron Firing Simulation", color="#f1f5f9", fontsize=14, fontweight="bold", ha="center")
        
        # Inputs
        c1 = "#38bdf8" if x[0] else "#1e293b"
        c2 = "#38bdf8" if x[1] else "#1e293b"
        ax.scatter([pos_x1[0]], [pos_x1[1]], s=800, color=c1, edgecolors="#475569", lw=2)
        ax.scatter([pos_x2[0]], [pos_x2[1]], s=800, color=c2, edgecolors="#475569", lw=2)
        ax.text(pos_x1[0], pos_x1[1], f"x1={x[0]}", color="white", ha="center", va="center", fontweight="bold", fontsize=10)
        ax.text(pos_x2[0], pos_x2[1], f"x2={x[1]}", color="white", ha="center", va="center", fontweight="bold", fontsize=10)
        
        # Travelling pulse
        prog = min(1.0, t_step / 6.0)
        if t_step <= 6:
            if x[0]:
                p = pos_x1 + (pos_soma - pos_x1) * prog
                ax.scatter([p[0]], [p[1]], s=150, color="#facc15", zorder=5)
            if x[1]:
                p = pos_x2 + (pos_soma - pos_x2) * prog
                ax.scatter([p[0]], [p[1]], s=150, color="#facc15", zorder=5)
                
        # Soma state
        soma_color = "#1e293b"
        soma_edge = "#475569"
        soma_size = 1500
        
        if t_step > 6:
            if fires:
                soma_color = "#15803d"
                soma_edge = "#4ade80"
                soma_size = 1800
                ring_r = 0.05 + 0.05 * ((t_step - 6) / 7.0)
                circ = plt.Circle(pos_soma, ring_r, color="#4ade80", fill=False, lw=2.5, alpha=max(0.1, 1 - (t_step-6)/7.0))
                ax.add_patch(circ)
            else:
                soma_color = "#7f1d1d"
                soma_edge = "#f87171"
                
        ax.scatter([pos_soma[0]], [pos_soma[1]], s=soma_size, color=soma_color, edgecolors=soma_edge, lw=3, zorder=3)
        ax.text(pos_soma[0], pos_soma[1]+0.015, "Soma", color="white", ha="center", va="center", fontweight="bold", fontsize=11, zorder=4)
        ax.text(pos_soma[0], pos_soma[1]-0.025, f"z={z:+.2f}", color="#facc15" if fires else "#fca5a5", ha="center", va="center", fontweight="bold", fontsize=9, zorder=4)
        
        # Output pulse
        out_col = "#1e293b"
        out_txt_col = "#64748b"
        out_val = 0
        if t_step > 7 and fires:
            prog_out = min(1.0, (t_step - 7) / 6.0)
            p_out = pos_soma + (pos_out - pos_soma) * prog_out
            ax.scatter([p_out[0]], [p_out[1]], s=180, color="#4ade80", zorder=5)
            if prog_out >= 0.8:
                out_col = "#15803d"
                out_txt_col = "#ffffff"
                out_val = 1
                
        ax.scatter([pos_out[0]], [pos_out[1]], s=900, color=out_col, edgecolors="#475569", lw=2, zorder=3)
        ax.text(pos_out[0], pos_out[1], f"Out={out_val}", color=out_txt_col, ha="center", va="center", fontweight="bold", fontsize=10, zorder=4)
        
        # Status text
        lbl = s["label"]
        if t_step <= 6:
            status = f"Input {lbl}: Signals Propagating along Synapses..."
            col = "#38bdf8"
        elif fires:
            status = f"Input {lbl}: THRESHOLD MET (z={z:+.2f} >= 0) -> FIRES! (Output = 1)"
            col = "#4ade80"
        else:
            status = f"Input {lbl}: SUB-THRESHOLD (z={z:+.2f} < 0) -> DORMANT (Output = 0)"
            col = "#f87171"
            
        ax.text(0.5, 0.08, status, color=col, fontsize=11, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e293b", edgecolor=col))
                
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        img = Image.fromarray(rgba).convert("RGB").convert("P", palette=Image.ADAPTIVE)
        frames.append(img)

plt.close(fig)
frames[0].save("neuron_firing.gif", save_all=True, append_images=frames[1:], duration=150, loop=0)
print(f"Successfully generated neuron_firing.gif with {len(frames)} frames!")
