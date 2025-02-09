# Fruit-Detect-YOLO9E
![val_batch2_pred](https://github.com/user-attachments/assets/b46bd058-7977-4670-923a-9a4eb687fbaf)
![P_curve](https://github.com/user-attachments/assets/34dd3aca-97f2-44f8-8e78-8fa83c547041)
![PR_curve](https://github.com/user-attachments/assets/034caf89-b8c4-43a9-8c21-b213c1b0e588)
![R_curve](https://github.com/user-attachments/assets/5e455dd4-41fe-411e-adca-d13f4ef43d88)
![F1_curve](https://github.com/user-attachments/assets/70d3259a-6c7f-44db-ae09-e57296308b7d)

# Fruit Detection using YOLOv9

This repository contains a project for detecting various fruits in images using the YOLOv9 object detection model. The system is designed to accurately locate and classify fruits such as Apple, Banana, Grapes, Kiwi, Mango, Orange, Pineapple, Sugerapple, and Watermelon.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This project leverages **YOLOv9** for real-time fruit detection. YOLOv9 was selected for its:
- **High Speed & Efficiency:** Capable of real-time processing with GPU acceleration.
- **High Accuracy:** Provides robust detection even in complex scenes.
- **Flexibility:** Easily adapts to different datasets and integrates advanced features like Automatic Mixed Precision (AMP).

For more details on YOLOv9’s features and improvements, visit [viso.ai's YOLOv9 page](https://viso.ai/computer-vision/yolov9/).

## Project Structure

```
.
├── data.yaml         # Dataset configuration file with paths and class names
├── train.py          # Script for training the YOLOv9 model
├── evaluate.py       # Script for evaluating the trained model
├── infer.py          # Script for performing inference on images or videos
├── utils.py          # Utility functions (e.g., for data extraction)
├── requirements.txt  # List of required Python packages
└── README.md         # This file
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/fruit-detection-yolov9.git
   cd fruit-detection-yolov9
   ```

2. **Install the required packages:**

   Make sure you have Python 3.11 (or later) installed, then run:
   
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Your Environment:**

   - Ensure you have a CUDA-enabled GPU (e.g., NVIDIA A100) for training and inference.
   - If using cloud platforms (e.g., Google Colab), adjust the file paths as necessary.

## Data Preparation

- **Dataset Format:**  
  The dataset is provided as a ZIP archive containing separate directories for training (`train`), validation (`valid`), and testing (`test`).

- **Extraction:**  
  Use the provided utility function in `utils.py` to extract the dataset:
  
  ```python
  extract_zip('path/to/dataset.zip', 'desired/extraction/path')
  ```

- **Configuration:**  
  Update the `data.yaml` file with the correct paths to your dataset folders. The file also lists the nine fruit classes.

## Training

To train the model, execute:

```bash
python train.py
```

**Training Details:**
- **Model:** YOLOv9e (an enhanced version of YOLOv9)
- **Epochs:** 50
- **Image Size:** 640×640
- **Batch Size:** 16
- **Optimizer:** AdamW (with automatic adjustment of learning rates and momentum)
- **Acceleration:** Utilizes Automatic Mixed Precision (AMP) for faster training

Training logs provide metrics such as box loss, classification loss, and DFL loss, as well as performance indicators (Precision, Recall, mAP50, mAP50-95).

## Evaluation

After training, evaluate the model on both validation and test sets using:

```bash
python evaluate.py --weights runs/detect/train8/weights/best.pt --data data.yaml
```

Typical evaluation metrics include:
- **Precision**
- **Recall**
- **mAP50:** Mean Average Precision at an IoU threshold of 50%
- **mAP50-95:** Mean Average Precision over multiple IoU thresholds

## Inference

Run the inference script to test the model on a specific image or video:

```bash
python infer.py
```

You will be prompted to select either "video" or "image" mode and then provide the corresponding file path. The script will display the results with annotated detections.

## Results

The trained model achieved the following performance on the test set:

- **Overall Metrics:**
  - Precision: ~0.77
  - Recall: ~0.71
  - mAP50: ~0.78
  - mAP50-95: ~0.65

- **Per-Class Performance:**
  - **Apple, Mango, Grapes:** High detection accuracy with excellent mAP scores.
  - **Banana:** Lower performance (e.g., lower recall) indicating the need for further fine-tuning or additional data.
  
- **Speed:**
  - Preprocessing: ~1.0ms per image
  - Inference: ~13.8ms per image
  - Postprocessing: ~1.5ms per image

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions, improvements, or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Ultralytics YOLOv9](https://github.com/ultralytics/ultralytics)
- [viso.ai](https://viso.ai/computer-vision/yolov9/) for the detailed insights on YOLOv9 enhancements.

---

Feel free to adjust the file paths and project details according to your specific setup. Happy coding!
