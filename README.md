# CAN Bus Anomaly Detection in Autonomous Vehicle Sensor Data

## Overview
This project focuses on detecting anomalies in the CAN (Controller Area Network) bus data collected from sensors in autonomous vehicles. The dataset includes sensor readings such as accelerometer RMS values, current, pressure, temperature, thermocouple measurements, voltage, and volume flow rate, which can be used to identify irregular patterns or malfunctions.

The dataset contains time-series data from various vehicle sensors, which is critical for real-time anomaly detection in safety-critical systems like autonomous vehicles.

## Dataset Description

### Columns:
- `tag`: Unique identifier for each data point.
- `datetime`: Timestamp when the data was recorded.
- `Accelerometer1RMS`: RMS value of the first accelerometer.
- `Accelerometer2RMS`: RMS value of the second accelerometer.
- `Current`: Current measurement in the vehicle's system.
- `Pressure`: Pressure reading from the sensors.
- `Temperature`: Internal vehicle temperature (ambient).
- `Thermocouple`: Thermocouple temperature measurement.
- `Voltage`: Voltage reading from the system.
- `Volume Flow RateRMS`: RMS value of the volume flow rate.

### Sample Data:
| tag | datetime           | Accelerometer1RMS | Accelerometer2RMS | Current | Pressure  | Temperature | Thermocouple | Voltage | Volume Flow RateRMS |
|-----|--------------------|-------------------|-------------------|---------|-----------|-------------|--------------|---------|---------------------|
| 1   | 2/8/2020 13:30     | 0.20603           | 0.277924          | 1.81019 | 0.382638  | 90.174      | 26.776       | 228.208 | 121.664             |
| 2   | 2/8/2020 13:30     | 0.204366          | 0.275727          | 2.66317 | -0.273216 | 90.2836     | 26.776       | 227.245 | 122                 |
| ... | ...                | ...               | ...               | ...     | ...       | ...         | ...          | ...     | ...                 |

## Objective
The primary goal is to develop a model that can identify anomalies in the sensor data, which could indicate hardware issues, environmental changes, or system failures. Anomalies are important to detect as they may lead to unsafe driving conditions in autonomous vehicles.

## Key Steps
1. **Data Preprocessing**: 
   - Handling missing values.
   - Scaling and normalizing sensor readings.
   - Feature engineering to extract additional insights from the raw data.
   
2. **Exploratory Data Analysis**:
   - Visualizing time-series data to understand normal vs. abnormal behavior.
   - Analyzing the distribution of various sensor readings.
   
3. **Anomaly Detection Techniques**:
   - **Unsupervised Learning**: Using clustering methods such as DBSCAN, Isolation Forest, or Autoencoders for detecting outliers in the data.
   - **Supervised Learning**: Training a model using labeled data to classify normal and anomalous patterns.

4. **Model Evaluation**:
   - Using precision, recall, F1-score, and AUC-ROC curves to evaluate the performance of the anomaly detection model.
   - Cross-validation and hyperparameter tuning for optimal results.

## Requirements
- Python 3.8+
- Required libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - tensorflow/keras (for deep learning models)
  - pyod (Python Outlier Detection Library)

## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/can-bus-anomaly-detection.git
    cd can-bus-anomaly-detection
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Load the dataset and preprocess the data:
    ```python
    from preprocessing import load_and_preprocess_data

    data = load_and_preprocess_data("data/can_bus_data.csv")
    ```

2. Train the anomaly detection model:
    ```python
    from model import train_model

    model = train_model(data)
    ```

3. Visualize and analyze the results:
    ```python
    from utils import plot_results

    plot_results(data, model)
    ```

## Results
The trained model will highlight anomalies in the dataset, which can then be analyzed to understand potential malfunctions or irregular patterns in the vehicle's sensor system.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
Special thanks to the contributors and the community working on autonomous vehicle systems, and the developers of open-source tools that made this project possible.

