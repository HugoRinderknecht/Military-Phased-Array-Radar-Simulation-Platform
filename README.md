# Military Phased Array Radar Simulation Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive phased array radar simulation platform implemented in Python, covering the complete simulation chain from target environment modeling, signal processing, and data processing to resource scheduling.

## Features

### Core Modules
- **Environment Simulation** - Multi-target environment with various motion models (CV, CA, CT, 6DOF)
- **Antenna System** - Phased array antenna modeling with beamforming and beam steering
- **Signal Processing** - Complete processing chain (LFM waveform, pulse compression, MTD, CFAR detection)
- **Data Processing** - Track initiation (M/N logic), data association, Kalman filtering
- **Resource Scheduling** - Adaptive task scheduling with multiple priority algorithms
- **Network Communication** - Real-time communication via FastAPI and WebSocket
- **Performance Evaluation** - Tracking accuracy, scheduling efficiency, detection metrics

### Technical Highlights
- **High Performance** - NumPy vectorization, Numba JIT compilation
- **Asynchronous Architecture** - Full asyncio-based I/O processing
- **Modular Design** - Clear interfaces, pluggable components
- **Complete Algorithms** - No simplification, fully implemented radar algorithms

## Technology Stack

- **Language**: Python 3.10+
- **Numerical Computing**: NumPy, SciPy, Numba
- **Web Framework**: FastAPI, WebSocket, Uvicorn
- **Data Processing**: scikit-learn, Pydantic
- **Testing**: pytest
- **Documentation**: Markdown, TOML configuration

## Project Structure

```
radar/
├── common/              # Common modules
│   ├── types.py         # Type definitions
│   ├── constants.py     # Physical constants
│   ├── utils/          # Utility functions
│   ├── containers/      # Data structures
│   ├── config.py        # Configuration management
│   └── logger.py       # Logging system
│
├── protocol/            # Communication protocol
│   ├── messages.py      # Message definitions
│   └── serializer.py   # Serialization
│
└── backend/             # Backend modules
    ├── core/            # Core control (time, state, radar_core)
    ├── environment/     # Environment simulation
    ├── antenna/         # Phased array antenna
    ├── signal/          # Signal processing
    ├── dataproc/        # Data processing & tracking
    ├── scheduler/       # Resource scheduling
    ├── network/         # Network communication
    └── evaluation/      # Performance evaluation
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/HugoRinderknecht/Military-Phased-Array-Radar-Simulation-Platform.git
cd Military-Phased-Array-Radar-Simulation-Platform
```

2. **Create virtual environment**
```bash
python -m venv .venv

# Activate on Linux/Mac
source .venv/bin/activate

# Activate on Windows
.venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Start the Backend Server

```bash
# Method 1: Using Python module
python -m radar.main

# Method 2: Using startup scripts
./start.sh     # Linux/Mac
start.bat       # Windows

# Method 3: Using Uvicorn (recommended for production)
uvicorn radar.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Access the API

Once the server is running:
- **HTTP API**: http://localhost:8000/api/status
- **WebSocket**: ws://localhost:8000/ws
- **API Documentation**: http://localhost:8000/docs (Swagger UI)

### Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_system.py

# Run with coverage
pytest --cov=radar --cov-report=html
```

### Run System Verification

```bash
python verify_system.py
```

## Configuration

Edit `radar_config.toml` to customize:
- Radar parameters (frequency, power, bandwidth)
- Antenna configuration (array geometry, beam patterns)
- Signal processing settings (waveform, detection thresholds)
- Data processing parameters (tracking filters, association gates)
- Scheduling parameters (priorities, time windows)
- Network settings (host, port, CORS)
- Logging configuration (level, file rotation)

## Examples

See `examples/simple_usage.py` for usage examples of all modules:

```python
import asyncio
from radar.backend.environment.simulator import EnvironmentSimulator
from radar.common.types import TargetType, Position3D, Velocity3D

async def main():
    # Create environment simulator
    env = EnvironmentSimulator()
    await env.initialize()

    # Add a target
    await env.add_target(
        target_type=TargetType.AIRCRAFT,
        position=Position3D(50000, 30000, 10000),
        velocity=Velocity3D(200, 150, 0),
        rcs=10.0
    )

    # Update simulation
    await env.update(0.1)

asyncio.run(main())
```

## Documentation

- [**Design Document**](docs/BACKEND_DESIGN.md) - System architecture and design
- [**Project Structure**](docs/PROJECT_STRUCTURE.md) - Detailed code organization
- [**Development Summary**](docs/DEVELOPMENT_SUMMARY.md) - Implementation status
- [**Final Report**](docs/FINAL_REPORT.md) - Complete feature list

## Performance

| Scenario | Targets | Frame Rate | CPU Usage | Memory |
|-----------|----------|------------|-----------|---------|
| Small     | 10       | 60 Hz      | ~10%      | ~200MB   |
| Medium    | 50       | 30 Hz      | ~30%      | ~500MB   |
| Large     | 100      | 20 Hz      | ~60%      | ~1GB     |

## Development Roadmap

- [x] Complete backend system (Phase 1)
- [ ] Frontend visualization interface
- [ ] GPU acceleration with CuPy
- [ ] Advanced motion models (Singer, Jerk)
- [ ] JPDA/MHT data association
- [ ] STAP (Space-Time Adaptive Processing)
- [ ] Clutter map implementation
- [ ] Electronic countermeasures (ECM)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Implemented based on modern radar system theory
- Reference: "Fundamentals of Radar Signal Processing" by Mark A. Richards
- Reference: "Principles of Radar" by Reintjes & Griffiths

---

**Military Phased Array Radar Simulation Platform** - A complete, production-ready radar simulation system.
