# P2P Energy Trading with Prospect Theory and Q-Learning

[![arXiv](https://img.shields.io/badge/arXiv-2208.12777-b31b1b.svg)](https://arxiv.org/abs/2208.12777)
[![Conference](https://img.shields.io/badge/Conference-IEEE%20GLOBECOM%202022-blue)](https://ieeexplore.ieee.org/document/10014205)

Source code for the paper: **"Prospect Theory-inspired automated P2P Energy Trading with Q-learning based Dynamic Pricing"** presented at IEEE GLOBECOM 2022.

## Abstract

The widespread adoption of distributed energy resources and smart grid technologies has enabled traditionally passive power system users to actively participate in energy trading. Traditional centralized grid-driven energy markets offer minimal profitability to these users, prompting a shift towards decentralized peer-to-peer (P2P) energy markets. In these markets, users trade energy directly with each other, yielding higher benefits compared to grid transactions.

However, existing P2P energy trading research often overlooks user perception, assuming constant availability, participation, and compliance. This can lead to negative attitudes and reduced engagement over time. This work designs an automated P2P energy market that incorporates user perception using prospect theory.

We formulate an optimization framework to maximize buyer's perceived utility while matching demand and production. Due to the non-linear and non-convex nature of the problem, we propose the **Differential Evolution-based Algorithm for Trading Energy (DEbATE)**. Additionally, we introduce a risk-sensitive Q-learning algorithm, **Pricing mechanism with Q-learning and Risk-sensitivity (PQR)**, which learns optimal prices for sellers considering their perceived utility.

Results, based on real energy consumption and production traces and realistic prospect theory functions, demonstrate that our approach achieves 26% higher perceived value for buyers and 7% more reward for sellers compared to state-of-the-art methods.

## Features

- **Prospect Theory Integration**: Models user perception in energy trading decisions.
- **DEbATE Algorithm**: Differential Evolution-based optimization for matching buyers and sellers.
- **PQR Algorithm**: Risk-sensitive Q-learning for dynamic pricing.
- **Comparative Analysis**: Benchmarks against rule-based and other state-of-the-art methods.
- **Real Data Simulation**: Uses realistic energy consumption and production data.

## Installation

### Prerequisites

- Python 3.7+
- Required Python packages: `numpy`, `pandas`, `networkx`, `seaborn`, `matplotlib`

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ashutoshtmlsna/P2P_energy_trading.git
   cd P2P_energy_trading
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas networkx seaborn matplotlib
   ```

   Note: If using a virtual environment, activate it before installing.

## Usage

### Configuration

The simulation parameters (number of sellers `s` and buyers `b`) are configured in `settings.py`. Modify the values of `s` and `b` to change the simulation scale.

Example configurations:
- `s = 5, b = 5`: Small-scale simulation
- `s = 10, b = 10`: Medium-scale simulation
- `s = 20, b = 20`: Large-scale simulation

### Running the Simulation

Execute the main script to run the simulations and generate results:

```bash
python main.py
```

The script will:
1. Run the rule-based baseline method.
2. Run the Zhu et al. method.
3. Run DEbATE with PQR pricing.
4. Run DEbATE with DQN pricing.
5. Generate plots comparing the objective values over time.

Results are saved as CSV files and plots are displayed.

### Output Files

- `obj_vals_rule_{s}_{b}.csv`: Objective values for rule-based method.
- `obj_vals_zhu_{s}_{b}.csv`: Objective values for Zhu method.
- `obj_vals_q_org_{s}_{b}.csv`: Objective values for DEbATE-PQR.
- `obj_vals_dqn_{s}_{b}.csv`: Objective values for DEbATE-DQN.
- Various average price and individual price CSV files.

## Algorithms

### DEbATE (Differential Evolution-based Algorithm for Trading Energy)

DEbATE uses differential evolution to solve the non-convex optimization problem of matching buyers and sellers while maximizing perceived utility under prospect theory.

### PQR (Pricing mechanism with Q-learning and Risk-sensitivity)

PQR employs Q-learning with risk-sensitive updates to learn optimal pricing strategies for sellers, considering their prospect theory-based utility functions.

### Baselines

- **Rule-based**: Simple matching rules.
- **Zhu et al.**: State-of-the-art P2P trading method.

## Data

The project uses the following data files:
- `consumer_predictions_150.csv`: Buyer consumption predictions.
- `producer_predictions_48.csv`: Seller production predictions.
- `pt_params_seller_50.csv`: Prospect theory parameters for sellers.
- `pt_params_buyer_150.csv`: Prospect theory parameters for buyers.
- `loss_45_150.csv`: Loss matrix for trading.

## Results

The simulation results show that DEbATE-PQR outperforms baselines in terms of buyer perceived value and seller rewards. See the paper for detailed analysis.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{timilsina2022prospect,
  title={Prospect Theory-inspired automated P2P Energy Trading with Q-learning based Dynamic Pricing},
  author={Timilsina, Ashutosh and others},
  booktitle={IEEE Global Communications Conference (GLOBECOM)},
  year={2022},
  pages={1--6},
  doi={10.1109/GLOBECOM48099.2022.10014205}
}
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Real energy data traces used in simulations.
- Prospect theory functions based on behavioral economics literature.
