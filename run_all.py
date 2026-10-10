"""
Master Runner for Neural Network Visual Learning Project
Allows launching all steps or the interactive web lab with a single command.
"""
import sys
import os
import subprocess
import webbrowser
import time

STEP_SCRIPTS = [
    ("step1_single_neuron.py", "Step 1: Single Neuron Basics"),
    ("step2_forward_pass_visual.py", "Step 2: Visualize Forward Pass"),
    ("step3_loss_function.py", "Step 3: Loss Functions (MSE & Cross-Entropy)"),
    ("step4_gradient_basics.py", "Step 4: Gradient Basics"),
    ("step5_simple_training.py", "Step 5: Simple Neuron Training (AND/OR Logic)"),
    ("step6_decision_boundary.py", "Step 6: Visualizing Decision Boundary"),
    ("step7_two_layer_network.py", "Step 7: Two-Layer Network (XOR Problem)"),
    ("step8_neuron_firing_animation.py", "Step 8: Live Neuron Firing Animation"),
]

def run_web_lab():
    print("=" * 60)
    print("🚀 Launching Neural Network Interactive Web Lab...")
    print("📍 URL: http://localhost:8000")
    print("=" * 60)
    try:
        webbrowser.open("http://localhost:8000")
    except Exception:
        pass
    subprocess.run([sys.executable, "server.py"])

def run_all_scripts():
    print("=" * 60)
    print("🧠 Running all 8 step-by-step learning scripts sequentially...")
    print("=" * 60)
    for filename, title in STEP_SCRIPTS:
        if not os.path.exists(filename):
            print(f"Skipping {filename} (not found)")
            continue
        print("\n" + "=" * 60)
        print(f"▶️  Running: {title} ({filename})")
        print("=" * 60)
        ret = subprocess.run([sys.executable, filename])
        if ret.returncode != 0:
            print(f"❌ Script {filename} exited with code {ret.returncode}")
            break
    print("\n✅ Completed all steps!")

def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ["--web", "-w", "web"]:
            run_web_lab()
            return
        elif arg in ["--scripts", "-s", "scripts", "--all", "-a", "all"]:
            run_all_scripts()
            return

    print("=" * 60)
    print("🧠 Neural Network Visual Learning - Master Runner")
    print("=" * 60)
    print("1) Launch Interactive Web Lab (Recommended - opens browser at http://localhost:8000)")
    print("2) Run All Step Scripts (step1 to step8 in order)")
    print("3) Run a specific step")
    print("0) Exit")
    print("=" * 60)
    try:
        choice = input("Enter choice [1-3, 0 to exit] (Default: 1): ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if choice == "" or choice == "1":
        run_web_lab()
    elif choice == "2":
        run_all_scripts()
    elif choice == "3":
        print("\nSelect a step to run:")
        for i, (fn, title) in enumerate(STEP_SCRIPTS, 1):
            print(f"  {i}) {title}")
        try:
            step_choice = input(f"Enter step number [1-{len(STEP_SCRIPTS)}]: ").strip()
            idx = int(step_choice) - 1
            if 0 <= idx < len(STEP_SCRIPTS):
                subprocess.run([sys.executable, STEP_SCRIPTS[idx][0]])
            else:
                print("Invalid step choice.")
        except Exception:
            pass
    elif choice == "0":
        print("Exiting.")

if __name__ == "__main__":
    main()
