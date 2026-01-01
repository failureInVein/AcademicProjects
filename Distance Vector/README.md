# Distance Vector Routing Algorithm in Python

A complete implementation of the **Distance Vector Routing Protocol** (Bellman-Ford) with real-time visualization, convergence detection, poison reverse, and dynamic link failure simulation.

This project simulates how routers exchange distance vectors, update routing tables, and converge to the shortest paths — exactly as used in early Internet routing protocols like RIP.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Requirements Met](#project-requirements-met)
- [Optional Features Included](#optional-features-included)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features
- Real-time animated network simulation using Pygame
- Bellman-Ford (Distance Vector) algorithm with **Poison Reverse**
- Automatic **convergence detection**
- Live routing table updates
- Dynamic **link failure and recovery** (press F/G)
- Packet animation showing distance vector exchange
- Beautiful, easy-to-understand GUI

## Installation

1. Make sure you have Python 3.8+ installed
2. Install the only dependency:
```bash
pip install pygame