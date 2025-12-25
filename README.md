# Hand Detection App - MediaPipe Hands

A professional hand detection application built with MediaPipe Hands AI, following clean architecture principles.

## Features

- Real-time hand detection using MediaPipe Hands
- Detection of up to 2 hands simultaneously
- 21 hand landmarks per hand with visual connections
- Handedness detection (Left/Right)
- FPS counter and hand count display
- Clean architecture with separation of concerns

## Architecture

The project follows clean architecture principles with three main layers:

```
handdetection/
├── domain/           # Core business logic and interfaces
│   ├── models.py     # Domain models (Hand, Point, DetectionResult)
│   └── interfaces.py # Abstract contracts (IHandDetector, ICamera)
├── data/             # External data sources and implementations
│   ├── mediapipe_detector.py  # MediaPipe Hands implementation
│   └── camera.py              # OpenCV camera implementation
├── presentation/     # UI and user interaction
│   └── hand_detection_viewer.py  # OpenCV viewer
├── main.py           # Application entry point
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

### Layer Responsibilities

- **Domain Layer**: Contains core business logic, models, and interfaces. No external dependencies.
- **Data Layer**: Implements domain interfaces using external libraries (MediaPipe, OpenCV).
- **Presentation Layer**: Handles user interface and visualization.

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
# On macOS/Linux, use pip3
pip3 install -r requirements.txt

# On Windows, use pip
pip install -r requirements.txt
```

**Note**: If you encounter permission errors, you may need to use:
```bash
pip3 install --user -r requirements.txt
```

## Usage

Run the application:
```bash
# On macOS/Linux, use python3
python3 main.py

# On Windows, use python
python main.py
```

The app will:
1. Open your default camera
2. Start detecting hands in real-time
3. Display hand landmarks and connections
4. Show FPS and hand count

Press `q` to quit the application.

## Requirements

- Python 3.8+
- Webcam/camera
- OpenCV
- MediaPipe
- NumPy

## Technical Details

- **Hand Landmarks**: 21 points per hand (wrist, thumb, index, middle, ring, pinky)
- **Detection Confidence**: 0.5 (configurable)
- **Max Hands**: 2 (configurable)
- **Frame Rate**: Optimized for real-time performance

## License

This project is open source and available for demonstration purposes.

