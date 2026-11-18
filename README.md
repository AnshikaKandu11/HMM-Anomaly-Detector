
# HMM-Based Anomaly Detection System

## 1. Project Overview

The HMM-Based Anomaly Detection System is a machine learning project designed to identify unusual patterns in sequential sensor data using a Hidden Markov Model (HMM).

The system is trained on normal machine sensor readings using Gaussian emissions. It calculates the log-likelihood of sensor data and identifies unusual readings using a threshold-based sliding window approach.

This project demonstrates how Hidden Markov Models can be used in predictive maintenance to detect abnormal machine behavior and help prevent unexpected equipment failures.

## 2. Objectives

- To understand and implement Hidden Markov Models.
- To train a Gaussian HMM using normal sensor data.
- To calculate log-likelihood scores.
- To detect anomalies using a sliding window approach.
- To visualize normal and anomalous sensor readings.
- To evaluate the performance of the anomaly detection system.

## 3. Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- hmmlearn
- Visual Studio Code

## 4. Project Structure

```text
HMM_Anomaly_Detector/
│
├── data/
│   ├── normal_data.csv
│   ├── test_data.csv
│   ├── detection_results.csv
│   └── anomaly_graph.png
│
├── generate_data.py
├── train_model.py
├── detect_anomalies.py
├── visualize.py
├── evaluation.py
├── realtime_detection.py
├── app.py
├── hmm_model.pkl
└── requirements.txt
```

## 5. Installation

First, install Python on your system.

Open the project folder in Visual Studio Code and open the terminal.

Install the required libraries:

```bash
pip install -r requirements.txt
```

## 6. How to Run the Project

Follow these steps to execute the project:

**Step 1:** Generate sensor data.

```bash
python generate_data.py
```

**Step 2:** Train the Hidden Markov Model.

```bash
python train_model.py
```

**Step 3:** Detect anomalies.

```bash
python detect_anomalies.py
```

**Step 4:** Generate the anomaly visualization.

```bash
python visualize.py
```

**Step 5:** Evaluate the model.

```bash
python evaluation.py
```

**Step 6:** Run real-time detection simulation.

```bash
python realtime_detection.py
```

**Step 7:** Run the main application.

```bash
python app.py
```

## 7. Working Methodology

1. Data Generation: Generate normal and test sensor readings.
2. Model Training: Train a Gaussian Hidden Markov Model using normal sensor data.
3. Likelihood Calculation: Calculate log-likelihood scores for sensor data windows.
4. Threshold Selection: Set an anomaly threshold using the likelihood distribution of normal training windows.
5. Anomaly Detection: Identify windows with likelihood scores below the threshold.
6. Visualization: Display sensor readings and detected anomalies using a graph.
7. Evaluation: Measure model performance using accuracy, precision, recall and F1-score.

## 8. Model Selection

The project uses a Gaussian Hidden Markov Model with three hidden states.

Three hidden states were selected to represent possible operating conditions in normal sensor data. Gaussian emissions are suitable because sensor readings are continuous numerical values.

The model is trained using the hmmlearn library.

## 9. Real-World Application

This system can be used in manufacturing industries for predictive maintenance.

It helps identify unusual machine behavior by analyzing sensor readings. Early detection of abnormal patterns can help maintenance teams investigate possible machine problems and reduce unexpected downtime.

## 10. Expected Output

- Generated normal and test sensor datasets.
- Trained Gaussian HMM model.
- Anomaly detection results in CSV format.
- Graph showing sensor readings and detected anomalies.
- Model evaluation report.
- Simulated real-time anomaly detection output.

## 11. Advantages

- Detects unusual patterns in sequential data.
- Uses probabilistic modeling.
- Supports sliding window-based detection.
- Provides graphical visualization.
- Can be applied to predictive maintenance.

## 12. Limitations

- The project uses simulated sensor data.
- Detection performance depends on the selected threshold.
- False positive detections may occur.
- Actual industrial sensor integration is not included.

## 13. Future Scope

- Integration with real-time industrial sensors.
- Development of a web-based monitoring dashboard.
- Improved threshold optimization.
- Testing with real industrial datasets.
- Sending alerts when anomalies are detected.

## 14. Conclusion

The HMM-Based Anomaly Detection System successfully demonstrates the application of Hidden Markov Models in sequential sensor data analysis.

By training the model on normal sensor readings and applying a sliding window approach, the system identifies unusual patterns in test data. The visualization and evaluation modules help understand the detection results.

This project provides practical knowledge of probabilistic machine learning and its application in predictive maintenance.

## 15. Author

**Project:** HMM-Based Anomaly Detection System  
**Programming Language:** Python  
**IDE:** Visual Studio Code
