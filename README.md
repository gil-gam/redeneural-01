# Guitar Level Classifier - Neural Network

A TensorFlow.js neural network that classifies guitar players into skill levels (beginner, intermediate, advanced) based on their technical knowledge.

> Part of the postgraduate project in **AI Applied Software Engineering**.

---

## Overview

This application demonstrates a practical implementation of machine learning in Node.js, using TensorFlow.js to build, train, and execute a classification model. The neural network analyzes guitar players' technical characteristics and predicts their skill level.

---

## Technical Stack

| Technology | Purpose |
|------------|---------|
| TensorFlow.js (v4.22) | Neural network framework |
| Node.js | Runtime environment |
| Docker + DevContainer | Reproducible development environment |

---

## Model ArchitectureInput Layer (9 features)
↓
Dense Layer (100 units, ReLU activation)
↓
Output Layer (3 units, Softmax activation)
**Features (One-Hot Encoded):**
- Chords: basic chords, CAGED, inverted chords
- Scales: pentatonic, natural minor, harmonic minor
- Licks: single shape, various shapes, various tones

**Classification Output:**
- Beginner
- Intermediate
- Advanced

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Epochs | 100 |
| Shuffle | Enabled |
| Metrics | Accuracy |

---

## Quick Start

### Prerequisites
- Node.js 18+
- Docker (optional, for containerized environment)

### Running Locally
```bash
# Install dependencies
npm install

# Run the application
npm startRunning with DevContainerOpen the project in VS Code with Dev Containers extension and select "Reopen in Container".

Example output:
beginner (99.87%)
intermediate (0.08%)
advanced (0.05%)

Project Structure
├── index.js              # Main application (train + predict)
├── package.json          # Dependencies and scripts
├── Dockerfile            # Container image definition
├── docker-compose.yml    # Docker Compose configuration
└── .devcontainer/        # VS Code DevContainer setup
    └── devcontainer.json

Neural network implementation with TensorFlow.js in Node.js
One-hot encoding for categorical features
Softmax activation for multi-class classification
Model training with real-time epoch logging
Containerized ML development environment

To add new training samples:
const tensorLevels = [
  [1, 0, 0, 1, 0, 0, 1, 0, 0], // beginner
  [1, 1, 0, 1, 1, 0, 1, 1, 0], // intermediate
  [1, 1, 1, 1, 1, 1, 1, 1, 1], // advanced
  // Add more samples here
];

const tensorLabels = [
  [1, 0, 0], // beginner
  [0, 1, 0], // intermediate
  [0, 0, 1], // advanced
  // Add corresponding labels
];

<p align="center">
  <img src="./images/imagejavascript.png" alt="JavaScript" width="60" />
  <img src="./images/imagenodejs.png" alt="Node.js" width="60" />
  <img src="./images/imagetensorflow.png" alt="TensorFlow" width="60" />
  <img src="./images/imagedocker.png" alt="Docker" width="60" />
</p>




