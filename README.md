# Neural Network Visual Learning 🧠

A beginner-friendly project to understand neural networks step-by-step with visual explanations, plots, and simple code.

## 🌐 Real Python & NumPy Interactive Web Lab

You can explore all steps interactively in your browser with real-time Python/NumPy execution:

```bash
python server.py
```
Then visit **[http://localhost:8000](http://localhost:8000)** in your browser. Every button click, slider movement, and training step is executed directly by Python and NumPy on your computer.

## Learning path

Follow these steps in order:

1. `step1_single_neuron.py` — learn weights, bias, and activation.
2. `step2_forward_pass_visual.py` — see data move through a neuron.
3. `step3_loss_function.py` — understand error/loss.
4. `step4_gradient_basics.py` — understand how learning direction is found.
5. `step5_simple_training.py` — train a neuron on AND/OR logic.
6. `step6_decision_boundary.py` — watch a decision boundary move.
7. `step7_two_layer_network.py` — understand hidden layers.
8. `step8_neuron_firing_animation.py` — live animation of signals propagating and neuron firing.

## Install

```bash
pip install numpy matplotlib
```

## Run

### Run everything with one command:
```bash
# Interactive menu or single command options:
python run_all.py            # Opens menu to run web lab or all scripts
python run_all.py --web      # Starts the backend server and opens the browser
python run_all.py --scripts  # Runs all 8 scripts sequentially (step 1 to step 8)
```

### Or run individually:
```bash
python step1_single_neuron.py
python step2_forward_pass_visual.py
python step3_loss_function.py
python step4_gradient_basics.py
python step5_simple_training.py
python step6_decision_boundary.py
python step7_two_layer_network.py
python step8_neuron_firing_animation.py
```

## What you will learn

- What a neuron is
- What weights and bias do
- What activation functions do
- How forward propagation works
- What loss means
- What gradients are
- How training updates weights
- How hidden layers make networks powerful

## Notes for beginners

- This project uses plain NumPy and Matplotlib.
- The goal is understanding, not speed.
- You will see a lot of printouts and plots.
- Run the files one by one. Do not skip.

## Goal

The goal is to make neural network ideas feel intuitive and visual before you move to frameworks like PyTorch or TensorFlow.
