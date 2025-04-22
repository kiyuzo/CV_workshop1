# Computer Vision Workshop 1: Face Detection and Recognition

This project implements face recognition using the Eigenfaces algorithm and a pre-trained pipeline. It uses OpenCV for webcam input and real-time face detection.

## Features

- Real-time face detection using Haar cascades
- Face recognition using PCA (Eigenfaces) and SVM classifier
- `MeanCentering` transformer included as part of preprocessing


## Requirements

Make sure the following Python packages are installed:

- `opencv-python`
- `numpy`
- `matplotlib`
- `scikit-learn`

To easily install the packages, copy the command below into vsc terminal

```bash
pip install opencv-python numpy matplotlib scikit-learn
```
## File Structure
```bash 
.
├── /.venv/                  # local virtual environment storing Python packages & dependencies (isolates from system Python)
├── /images/                 # dataset images, you may add or remove (each person uses 10 different pictures)
├── CV_Workshop1.ipynb       # notebook file
├── README.md                # hey! you're currently reading this :>
├── dataset.zip              # downloaded automatically from gdrive
├── eigenface_pipeline.pkl   # saved trained model
└── script.py                # the main app used to run the program.
```

## How To Run
1. Ensure the model file exists
Make sure the file eigenface_pipeline.pkl (your pre-trained pipeline) exists and it is in the same directory as script.py.

2. Run the script app
```bash
python script.py
```

3. Using the Application

The webcam will open and start detecting faces.

Recognized faces will be labelled with names and scores in real-time.

4. Exit the Application
Press q on your keyboard to quit the webcam and close the application.

## Example Output
[![Watch the video](https://img.youtube.com/vi/pW91ymKl44k/0.jpg)](https://youtu.be/pW91ymKl44k)

## Notes
- The script includes a custom transformer class called MeanCentering, which must be defined in the same file when loading the pipeline using pickle.

- The model must be trained beforehand and serialized to eigenface_pipeline.pkl.

- To modify the model, you can edit the notebook directly or using google colab, then switch the old pickle file with the new one. If extra features or methods were added, you might have to modify the script.py accordingly.

This project is developed as a part of a Workshop Assignment for my Computer Vision Class