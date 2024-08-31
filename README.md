# YOLO8 Object Detection with MMYolo and Docker

This repository contains instructions for performing object detection tasks by training an YOLO8 model using MMYolo with Docker.

## Prerequisites

Before you begin, ensure you have Docker installed on your system. If not, you can install it by following the instructions [here](https://docs.docker.com/get-docker/).

## Getting Started

Clone this GitHub repository:

```bash
git clone https://github.com/javad-rezaie/YOLO8
cd YOLO
```

## Setting Up the Docker Environment


Build Docker image by running:

```bash
make docker-build-mmyolo
```

## Dataset Preparation

1. Download the "COCO-Stuff 10K dataset v1.1" dataset from [cocostuff10k GitHub Repo](https://github.com/nightrome/cocostuff10k?tab=readme-ov-file).
2. Unzip the dataset and place it in a suitable directory.
3. Convert the dataset format to COCO format by following the steps described in `Data_Preparation.ipynb` Jupyter notebook.

To run the Jupyter notebook from the terminal, execute the `jupyter.sh` script:

```bash
bash jupyter.sh
```

## Modifying the Paths and GPU Configuration

1. Update the `DATA_DIR` path inside the `train.sh` script to your appropriate local path where the dataset is located.
2. Update the `GPU` variable to the number of installed GPUs on your PC.

### Data Structure

#### Local Machine

On your local computer, the data structure is as follows:

/mnt/SSD2/coco_stuff10k/ ├── images/ ├── train_coco.json └── test_coco.json


#### Container

Within the container, this directory is accessible as `/data` and will appear as:

/data/ ├── images/ ├── train_coco.json └── test_coco.json


The local path `/mnt/SSD2/coco_stuff10k/` is mapped to `/data/` inside the container.

## Tips
Ensure that the `train.sh` and  `jupyter.sh` bash scripts has executable permissions. If not, grant execute permission by running `chmod u+x train.sh`.

# Model Conversion to OpenVINO and Hugging Face Integration
## Converting to OpenVINO
Our trained PyTorch model was converted to OpenVINO format using the Model Optimizer tool. This streamlined the deployment process for various hardware platforms.

## Hugging Face Upload
We shared the trained models (original PyTorch model and its converted version to OpenVINO format) on the Hugging Face Model Hub, making it easily accessible for developers ([here](https://huggingface.co/spaces/homai/YOLOv8-Real-Time-Object-Detection)). This allows for straightforward integration into applications and fine-tuning on custom datasets.

## Running on Hugging Face
Instantiating the model from its unique identifier on Hugging Face enables easy execution and result visualization. Whether through the website interface or the API, running the model is intuitive and efficient.

## Disclaimer

This project is intended for educational purposes only. It is not intended to provide medical advice or any other professional advice. Any use of this project for real-world applications should be done with caution and proper consultation with relevant experts.

## License

This project is licensed under the This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
