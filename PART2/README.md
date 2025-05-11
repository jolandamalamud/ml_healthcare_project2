# PART2

Part 2 implements a CNN for pneumonia detection from chest X-ray images, including model training, evaluation, and interpretability techniques.

## Project Structure

```
├── Q1_EDA.ipynb                      # Data exploration and preprocessing
├── Q2_CNN.ipynb                      # CNN model implementation and training
├── Q3_Integrated_Gradients.ipynb     # Model interpretability using Integrated Gradients
├── Q4_GradCAM.ipynb                  # Model interpretability using GradCAM
├── Q5_Data_Randomization_Test.ipynb  # Sanity checks using randomized data
├── pneumoniacnn.py                   # CNN model class
├── utils.py                          # Helper functions
├── requirements.txt                  # Dependencies
└── figs/                             # Generated figures
```

## Requirements

Install the necessary packages with:

```bash
pip install -r requirements.txt
```

## Note

Due to size limitations, the data and model weights (cnn/*.pth) are not included. 
