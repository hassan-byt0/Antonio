# AirSim Autonomous Driving RL+GNN+Logic Framework

This project presents a complete framework for developing and training an autonomous driving agent using a hybrid approach that combines Deep Reinforcement Learning (DRL), Graph Neural Networks (GNNs), and a logic-based rule system. The system is designed to integrate with the Microsoft AirSim simulator for realistic, simulated environments.

The core idea is to train a DRL agent to learn a robust driving policy while a GNN-based module ensures the agent adheres to complex traffic rules and social norms.

## Features

- **AirSim Integration**: Seamless communication with the Microsoft AirSim simulator to control a simulated car and receive sensor data (RGB, depth, segmentation)
- **Deep Reinforcement Learning (DQN)**: An agent trained using the Deep Q-Network (DQN) algorithm to learn an optimal policy for navigating the environment
- **Graph Neural Network (GNN)**: A GNN is included to reason about the dynamic relationships between the car and other agents or traffic elements, providing a layer for traffic rule compliance
- **Hybrid Architecture**: Combines the reactive, learned policy from the DQN with the rule-based, logical reasoning of the GNN
- **Training & Testing Modes**: The framework supports distinct modes for training the agent and for evaluating its performance
- **Model Persistence**: Ability to save and load trained models for continuous development and deployment

## Prerequisites

Before running the code, ensure you have the following installed and configured:

- **AirSim Simulator**: The Microsoft AirSim plugin for either Unreal Engine or Unity. You must have a simulated environment running and ready to accept connections
- **Python 3.x**
- **Required Python Libraries**: Install the necessary packages using pip. The requirements.txt file lists all dependencies

```bash
pip install airsim torch torch-geometric opencv-python numpy
```

## Installation and Usage

### Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Start the AirSim Environment

Launch your chosen AirSim environment (e.g., the Neighborhood map in Unreal Engine).

### Run the Framework

The main script can be run in two modes: `TRAIN` or `TEST`. You can configure the mode and other hyperparameters in the `Config` class at the top of the `main.py` script.

#### Training Mode

To train a new model, set `mode=Mode.TRAIN` in the `Config` dataclass and run the script. The agent will begin exploring and learning, and a model file will be saved.

```bash
python main.py
```

#### Testing Mode

To test a pre-trained model, set `mode=Mode.TEST` and `load_model=True` in the `Config` dataclass. Ensure the model file path in `model_path` is correct.

```python
config = Config(
    mode=Mode.TEST,
    load_model=True,
    model_path="model.pth"
)
```

Then, run the script:

```bash
python main.py
```

## Code Structure

- **Config**: A dataclass for managing all the training and simulation hyperparameters
- **AirSimClient**: A wrapper class to handle all communication and data retrieval from the AirSim simulator
- **DQNAgent**: Contains the core of the Reinforcement Learning agent, including the neural network (`_build_model`), experience replay memory, and methods for action selection and learning (`act`, `replay`)
- **TrafficRuleNet**: A placeholder for the Graph Neural Network that would process a graph representation of the environment to enforce traffic rules
- **AutonomousDrivingEnv**: Acts as the environment, handling the agent's interaction with the simulator and calculating rewards
- **train(config)**: The main function for the training loop
- **test(config)**: The main function for the evaluation loop
- **main()**: The entry point for the application

## Contributing and Future Work

This project provides a foundational framework. We welcome contributions to improve and expand its capabilities. Some potential areas for development include:

- **Refining the Reward Function**: Implementing a more complex and accurate reward function to guide the agent more effectively
- **Advanced GNNs**: Developing a robust GNN model that can parse and reason about a more sophisticated scene graph from the simulator data
- **Multi-Agent Interaction**: Extending the framework to handle interactions with other vehicles and pedestrians
- **Sensor Fusion**: Incorporating more sensor data (e.g., LiDAR, radar) for a richer state representation

## License

[Add your license information here]

## Contact

[Add your contact information here]
