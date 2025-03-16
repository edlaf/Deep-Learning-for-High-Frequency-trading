--- REINFORCEMENT LEARNING FOR HIGH FREQUENCY TRADING ---

The architecture of the project is the following:

project/
├── backend
│   ├── Market_env
│   │   └── Market.py
│   ├── QRModel
│   │   ├── QR_agent.py
│   │   # Implentation of the QR describe in our paper with the agent's interaction
│   │   └── QR_only.py
│   │   # Implentation of the QR describe in our paper without the agent's interaction
│   ├── RL_agents
│   │   ├── CA.py
│       # Implementation of the Actor-Critic Algorithm
│   │   ├── PPO.py
│       # Implementation of the PPO Algorithm
│   │   └── QDRL.py
│       # Implementation of the Deep Q-Reinforcement Learning Algorithm
│   └── utils
│       └── intensity_fct_params.py
├── Notebooks
│   ├── main_2.ipynb
│   └── main.ipynb
│   # This is wherre the main visualisation of the code is done
└── README.md

The paper relating our work can be find in papers.