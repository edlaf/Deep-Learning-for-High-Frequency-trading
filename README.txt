--- REINFORCEMENT LEARNING FOR HIGH FREQUENCY TRADING ---

The architecture of the project is the following:

projet/
├── Notebooks/
│   └── main.ipynb
|       # This is wherre the main visualisation of the code is done
├── backend/
│   ├── QRModel/
│   │   └── QR_only.py
│   │   # Implentation of the QR describe in our paper without the agent's interaction
│   │   └── QR_agent.py
│   │   # Implentation of the QR describe in our paper with the agent's interaction
│   └── RL_agents/
│       └── QDRL.py
│       # Implementation of the Deep Q-Reinforcement Learning Algorithm
└── utils/
    └── intensity_fct_params.py
    # Parameters set for the QR in the whole simulation

The paper relating our work can be find in papers.
