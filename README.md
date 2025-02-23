# Image-Colorization-Using-GANs

## Project Overview
This project focuses on colorizing grayscale images using **Generative Adversarial Networks (GANs)**. The primary goal is to build a model that can automatically add realistic colors to black-and-white photos, enhancing their visual appeal while preserving the original details.

## Objectives
- To develop a GAN-based model that effectively colorizes grayscale images.
- To train the model on a dataset of colored images and evaluate its performance using qualitative and quantitative metrics.
- To implement and fine-tune the architecture for optimal colorization results.

## Features
- Utilizes **Generative Adversarial Networks (GANs)** for image colorization.
- Efficient handling of dataset using automated downloading and extraction from Google Drive.
- Custom architecture design to improve colorization quality and model accuracy.
- Realistic colorization while preserving the original details of grayscale images.

## Installation and Requirements
Make sure you have the following libraries installed before running the project:

- Python 3.x
- TensorFlow or PyTorch (Choose based on implementation)
- NumPy
- OpenCV
- Matplotlib
- gdown (For downloading datasets from Google Drive)

To install the required libraries, run:
```bash
pip install numpy opencv-python-headless matplotlib gdown tensorflow
```

## Dataset
The dataset is downloaded from a shared Google Drive link using `gdown`. Since the project is executed on a **Vast.ai GPU instance**, direct access to Google Drive is not available, and thus `gdown` is used for fetching the ZIP file.

The dataset is then extracted programmatically, making it easier to handle large archives and automate the workflow for image colorization.

## Usage

1. **Download Dataset**:
   The script automates the download of the dataset from a Google Drive link using `gdown`.

2. **Extract Dataset**:
   The ZIP file is extracted programmatically for ease of use and automated workflow.

3. **Model Training**:
   - Train the GAN model on the extracted dataset of grayscale and colored image pairs.
   - Evaluate the model using both qualitative (visual inspection) and quantitative metrics (e.g., PSNR, SSIM).

4. **Image Colorization**:
   - Input: Grayscale images
   - Output: Colorized images with realistic and vivid colors.

## Running the Project
To run the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/image-colorization-gan.git
    cd image-colorization-gan
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Download and extract the dataset:
    ```python
    !gdown --id YOUR_DRIVE_FILE_ID
    !unzip dataset.zip -d ./data
    ```

4. Train the GAN model:
    ```python
    python train.py
    ```

5. Test the model:
    ```python
    python test.py
    ```

## Results and Evaluation
The performance of the model is evaluated using:
- **Qualitative Metrics**: Visual inspection of the colorized images.
- **Quantitative Metrics**:
  - **PSNR (Peak Signal-to-Noise Ratio)**: Measures the similarity between the original and colorized images.
  - **SSIM (Structural Similarity Index)**: Evaluates the structural similarity between images.


## Acknowledgments
- We would like to acknowledge **Vast.ai** for providing the GPU instance for model training.
- Dataset courtesy of publicly available resources on **Google Drive**.
