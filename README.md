# Real-Time Emotion Recognition from Facial Expressions Using Convolutional Neural Networks

## Overview
https://github.com/farukaydogan/Real-Time-Emotion-Recognition-from-Facial-Expressions/assets/57232389/469fac16-5c33-44e5-9ed9-915cb3ff7990

This repository contains the implementation of a real-time emotion recognition system that utilizes Convolutional Neural Networks (CNN) to analyze facial expressions. This system demonstrates the power of deep learning techniques in the field of computer vision.
## Prerequisites

- Matlab 2019b or above
- MATLAB Support Package for USB Webcams

## Implementation 

Details

The system works by first capturing real-time facial expressions using a webcam. The facial images are then processed by the CNN, which has been trained to recognize different emotions based on facial expressions, thus identifying the subject's emotion in real time.

The CNN model was trained on an improved version of the FER2013 dataset, which has been carefully labeled for different emotional states. Each emotion corresponds to certain unique facial expressions, and our CNN model has been trained to identify these features with high accuracy.
<img width="560" alt="Screenshot 2023-06-06 at 00 05 38" src="https://github.com/farukaydogan/Real-Time-Emotion-Recognition-from-Facial-Expressions/assets/57232389/e5c42185-50c8-4446-afb8-33d84474e758">

## Getting Started

### Installation

1. Clone the repository: git clone https://github.com/farukaydogan/Real-Time-Emotion-Recognition-from-Facial-Expressions
2. Navigate to the repository: cd Gender-Detection-On-Realtime-Voice
3. Open MATLAB and run the main file.


### [Dataset](https://drive.google.com/file/d/1yuUOEh4RyFF5KOPEpNX1K4l6JESv41vG/view?usp=sharing)

### Usage

After running the main file, the system will start processing real-time audio data from your computer's default microphone. The gender of the speaker will be outputted in real-time.

## Results and Future Work

The system has shown impressive accuracy rates during preliminary tests. However, as with all machine learning projects, there is always room for improvement. Future efforts will be directed towards refining the model to increase its accuracy and exploring its potential in other applications.

## Contributing

Contributions are always welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any queries or discussions, feel free to open an issue.
