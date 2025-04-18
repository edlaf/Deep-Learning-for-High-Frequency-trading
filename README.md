# Reinforcement Learning for High Frequency Trading

## Introduction
This project explores the application of **Reinforcement Learning (RL) for High-Frequency Trading (HFT)**. We develop a simulated market environment and train RL agents using **Deep Q-Learning (DQL), Actor-Critic (AC), and Proximal Policy Optimization (PPO)** algorithms. The goal is to design an agent capable of making optimal trading decisions under a realistic market simulation.

---

## Project Structure
The project is organized as follows:

```
project/
├── backend
│   ├── Market_env
│   │   └── Market.py
│   ├── QRModel
│   │   ├── QR_agent.py       # Implementation of QR model with agent interaction
│   │   └── QR_only.py        # Implementation of QR model without agent interaction
│   ├── RL_agents
│   │   ├── CA.py             # Implementation of the Actor-Critic Algorithm
│   │   ├── PPO.py            # Implementation of the PPO Algorithm
│   │   └── QDRL.py           # Implementation of the Deep Q-Reinforcement Learning Algorithm
│   └── utils
│       └── intensity_fct_params.py
├── Notebooks
│   ├── main_2.ipynb
│   └── main.ipynb            # Jupyter Notebooks for visualization and testing
└── README.md
```

---

## Implementation Details

### Environment Design
- We simulate a **limit order book (LOB)** based on the **Queue-Reactive (QR) model**.
- The order book includes **order additions, cancellations, and executions** modeled via Poisson processes.
- The environment captures **market impact** and price dynamics through stochastic processes.

### Agent Design
- **Deep Q-Learning (DQL)** agent is implemented using PyTorch.
- **State representation:** Market features such as bid/ask levels, time, price, and agent position.
- **Action space:** Buy, Sell, or Do Nothing.
- **Reward function:** Based on Profit and Loss (P&L) evolution.
- Experience replay and target network stabilization are used to improve learning.

### Alternative Approaches
- **Actor-Critic (AC)**: Combines a policy (actor) and value estimation (critic) for policy optimization.
- **Proximal Policy Optimization (PPO)**: Uses a clipped surrogate objective to ensure stable updates.
- Both methods were tested but exhibited **sensitivity to hyperparameters** and suboptimal policy convergence in some scenarios.

---

## Results
- The DQL agent **outperforms random strategies** in terms of profitability.
- The model **learns to balance aggressive and patient trading** behaviors over time.
- AC and PPO **require fine-tuning** to avoid local optima.

### Limitations & Future Work
- **Market realism**: The model currently simplifies certain aspects such as **latency, transaction costs, and order matching**.
- **Computational efficiency**: Training times can be improved by **parallelization and GPU acceleration**.
- **Scalability**: Extending the model to **multi-agent** scenarios or integrating more sophisticated trading strategies.

---

## Installation & Usage
### Requirements
- Python 3.8+

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/edlaf/Deep-Learning-for-High-Frequency-trading.git
   cd Deep-Learning-for-High-Frequency-trading
   ```
2. Create a virtual environment (optional but recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Visualize results in Jupyter Notebook:
   ```bash
   jupyter notebook Notebooks/main.ipynb
   ```

---

## References
- **Queue-Reactive Model**: [M. Rosenbaum, C.-A. Lehalle](https://arxiv.org/pdf/2003.10823.pdf)
- **Deep Q-Learning**: [Mnih et al., 2015](https://arxiv.org/abs/1312.5602)
- **PPO Algorithm**: [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)

---

## Contributors
- Edouard Laferté
- Paul Le Van Kiem
- Théo Le Pendeven
- David Kerriou
- Erwin Poussi

---
