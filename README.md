# Flight Delays Prediction Using Machine Learning
 

## Overview
This project focuses on predicting flight delays using machine learning techniques. By analyzing historical flight data, weather conditions, and airline-specific parameters, we aim to develop a predictive model that enhances operational decision-making and improves passenger experience.

## Authors
- **Saeed Sameer Al-Hasan**
- **Hussein Riyad Abu Anzeh**
- **Osama Tayseer Abu Alwafa**

## Supervisors
- **Dr. Ruba Obiedat**
- **Dr. Ali Al Rodan**

## University Affiliation
- **Data Science Department, King Abdullah II School of Information Technology, The University of Jordan**
- **Date: April 2024**

## Problem Statement
Flight delays impact airlines financially and affect passenger experience. Delays are caused by various factors such as weather conditions, security issues, and technical difficulties. This project aims to develop a machine learning model that accurately predicts flight delays based on historical data.

## Objectives
1. Develop a high-accuracy machine learning model to predict flight delays.
2. Utilize historical flight data, weather conditions, and operational parameters for better predictions.
3. Create a user-friendly graphical user interface (GUI) to make predictions accessible.

## Methodology
### Data Processing & Feature Engineering
- **Dataset:** Includes 28,820 flight records with weather and operational attributes.
- **Preprocessing:** Handling missing values, duplicate data, and formatting inconsistencies.
- **Feature Engineering:** Identifying the most relevant parameters affecting flight delays.

### Machine Learning Models
We experimented with several algorithms, including:
- Logistic Regression
- Random Forest
- Decision Tree (Best Performing Model: Accuracy = 96.84%)
- Support Vector Machine (SVM)
- k-Nearest Neighbors (k-NN)
- Convolutional Neural Networks (CNN)

### Model Evaluation
- **Metrics Used:** Accuracy, Precision, Recall, F1 Score.
- **Best Model:** Decision Tree classifier with an accuracy of 96.84%.

## Implementation
1. **Data Collection & Cleaning**: Preprocessing flight and weather data.
2. **Feature Selection**: Identifying key predictors for delays.
3. **Model Training & Validation**: Using scikit-learn to train and evaluate models.
4. **GUI Development**: Built an interactive GUI to predict delays.

## Results & Findings
- The Decision Tree model performed the best, achieving high accuracy in delay prediction.
- Synthetic Minority Over-sampling Technique (SMOTE) was used to balance data and improve model performance.
- A user-friendly GUI was implemented to allow easy interaction with the prediction model.

## Strengths & Limitations
### Strengths
- Comprehensive analysis incorporating multiple flight and weather features.
- High model accuracy using advanced ML techniques.
- Balanced dataset using SMOTE for fairer predictions.

### Limitations
- Limited dataset scope; model performance may vary across different regions and airlines.
- No real-time integration; future work should focus on live data streaming.

## Future Work
1. **Real-Time Predictions**: Integrate with live flight tracking systems.
2. **Enhanced Feature Engineering**: Incorporate additional external factors such as air traffic congestion.
3. **Optimization Techniques**: Improve model efficiency and generalization.

## How to Use
### Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

### Running the Model
```bash
python flight_delay_prediction.py
```

### Using the GUI
```bash
python gui.py
```

## References
1. Research papers and methodologies referenced in the full project report.
2. External sources on flight delay predictions and machine learning best practices.

## License
This project is open-source and available for academic and research purposes.

