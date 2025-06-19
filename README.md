# ONNX Runtime Sample Apps for Qualcomm Hexagon NPU
This repository contains sample apps for running ONNX models efficiently using [ONNX Runtime](https://onnxruntime.ai/), specifically targeting Qualcomm Hexagon NPU with [QNN Execution Provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
## Table of Contents
1. [Available Apps](#available-apps)
2. [Getting Started](#getting-started)
3. [Quick Start](#quick-start)
4. [Contributing](#contributing)
5. [Testing](#testing)
6. [Directory Structure](#directory-structure)
7. [License](#license)

## Available Apps
| App Name | Model Used | Providers | Quick Start | Notebook | Blog | Video |
|----------|------------|-----------|-------------|----------|-------|------|
| HRNet Pose Detection | HRNetPose | CPU | [Complete](#quick-start) | [Complete](./notebooks/pose_detection/) | [Blog](https://www.qualcomm.com/developer/blog/2025/03/enable-pose-detection-snapdragon-x-elite-step-by-step-tutorial) | [Youtube](https://youtu.be/OASSOhlSpfY?si=gNJLRHAxpl4IUflv) |
| DeepSeek Local | DeepSeek | QNN/CPU | [Complete](#quick-start) | [Complete](./notebooks/llm/) | 🚧 Coming Soon  | 🎬 Coming Soon |

These apps demonstrate end-to-end inference using ONNX Runtime on devices with Hexagon NPUs. Each app includes:
- Input preprocessing
- Onnx model inference
- Output postprocessing
  
Supported features:
- CPU fallback (if you don't have access to Hexagon NPU)
- Hexagon QNN Execution

## Getting Started
#### General Requirements
- Python (version 3.11.+)
   - If targeting Hexagon ensure you install ARM64 compatible Python version
- C and C++ support in Visual Studio
   - [Installation Instructions](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170)
- Rust Support
   - [Installation Instructions](https://rustup.rs/)
 - Git Support
   - [Installation Instructions](https://git-scm.com/downloads/win)
#### 1. Clone the Repository
```
git clone https://github.com/DerrickJ1612/qnn_sample_apps.git
```
#### 2. Setup Virtual Environment
```
python -m venv venv
venv\Scripts\activate.ps1 
pip install -r \src\<App Name>\requirements.txt # Ex: pip install -r \src\deepseek_r1\requirements.txt
```
#### 3. Download Models
| Model Name  | Description              | Download Source                                                                                                                                                                                                            |
|-------------|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| HRNetPose   | Human pose estimation    | [AI Hub](https://aihub.qualcomm.com/compute/models/hrnet_pose?domain=Computer+Vision&useCase=Pose+Estimation)                                                                                                              |
| DeepSeek R1 | Reasoning Language Model | [Microsoft AI Toolkit](https://marketplace.visualstudio.com/items?itemName=ms-windows-ai-studio.windows-ai-studio); [Google Drive](https://drive.google.com/drive/folders/1hCopYw7rMdeOm3zV6NC2do9orzpKqAMf?usp=sharing)   | 

## Quick Start

| App Name               | CLI Command                                  |
|------------------------|----------------------------------------------|
| 'HRNet Pose Detection' | ` >> python ./src/hrnet_pose/main.py `       |
| 'DeepSeek Local'       | ` >> python ./src/deepseek_r1/main.py --help` |

## Contributing
We welcome contributions to this repository! Please refer to our [contributing guide](CONTRIBUTING.md) for how to contribute.

## Testing
- All regression tests must pass
- New features should include appropriate test coverage
```
cd ./qnn_sample_apps
pytest -vv
```
## License
This project is licensed under the [MIT](https://github.com/DerrickJ1612/qnn_sample_apps/blob/main/LICENSE.txt)

## Directory Structure
```
QNN_SAMPLE_APPS/
├── models/
│   ├── cpu-deepseek-r1-distill-qwen-7b/
│   ├── hrnet_pose/
│   ├── qnn-deepseek-r1-distill-qwen-1.5b/
│   ├── qnn-deepseek-r1-distill-qwen-7b/
│   ├── qnn-deepseek-r1-distill-qwen-14b/
├── notebooks/
│   ├── llm/
│   ├── pose_detection/
├── scripts/
├── src/
│   ├── deepseek_r1/
│   ├── hrnet_pose/
│   └── model_loader.py
├── tests/
├── CODE_OF_CONDUCT.md
├── conftest.py
├── CONTRIBUTING.md
├── desktop.ini
├── executioner.json
├── LICENSE.txt
├── models.json
├── pyproject.toml
├── README.md
└── setup.py
```


