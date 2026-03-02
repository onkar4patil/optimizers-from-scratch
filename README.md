# Optimization Algorithms from Scratch: Momentum, RMSProp, Adam vs Batch Gradient Descent

Educational implementations of Momentum, RMSProp, and Adam, compared against plain batch gradient descent (BGD) on simple regression/classification toy problems.

## Motivation / Concepts

### Why plain Batch Gradient Descent can be slow

- Computes the gradient over the **entire dataset** for every update, which is expensive for large datasets.  
- Often requires a **small learning rate** to remain stable, so convergence can take many iterations.  
- Can get **stuck or oscillate** in narrow valleys or around plateaus.

### Stochastic Gradient Descent (SGD)

- Instead of using all data each step, SGD uses **a single example or a mini‑batch** to estimate the gradient.  
- This makes updates **much cheaper per step** and introduces noise that can help escape shallow local minima or plateaus.  
- Converges faster in wall‑clock time on large datasets, at the cost of a noisier optimization path.

### Momentum Gradient Descent

- Plain SGD/BGD can **zig‑zag** and slow down in ravines where the gradient changes direction frequently.  
- Momentum accumulates a **velocity term**: updates depend on both the current gradient and a fraction of the **previous update**.  
- This helps the optimizer build speed in consistent directions and **smooths out oscillations**, often converging faster.

### RMSProp

- Different parameters can have very different gradient scales, so a single global learning rate is not ideal.  
- RMSProp keeps a **running average of squared gradients** per parameter and divides the learning rate by the square root of this average.  
- The result is an **adaptive learning rate** per parameter that tends to stabilize and speed up training.

### Adam Optimizer

- Adam combines ideas from **Momentum** (moving average of gradients) and **RMSProp** (moving average of squared gradients).  
- Maintains both a **first moment (mean)** and **second moment (variance)** estimate for each parameter and uses bias‑correction.  
- Works well “out of the box” on many problems, with relatively little manual learning‑rate tuning.

## Repository Structure

All scripts are in the top‑level directory:

- `sg_vs_bgd.py` – compares stochastic gradient descent vs batch gradient descent.  
- `moment_vs_bgd.py` – compares Momentum vs BGD.  
- `rmsprop_vs_bgd.py` – compares RMSProp vs BGD.  
- `adam_vs_bgd.py` – compares Adam vs BGD.

## How to Run

From the project root:

```bash
python sg_vs_bgd.py
python moment_vs_bgd.py
python rmsprop_vs_bgd.py
python adam_vs_bgd.py
