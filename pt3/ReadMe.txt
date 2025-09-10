Accident Data Analysis and Severity Prediction System

Overview:
-This project is a comprehensive accident data analysis and severity prediction system built with Python. It uses machine learning (Random Forest Classifier) to predict accident severity based on various environmental and situational factors, and provides interactive visualizations to analyze accident patterns.

Features:
Data Visualization: Interactive plots showing accident patterns by weather conditions, time of day, number of vehicles involved, and road surface conditions
Severity Prediction: Machine learning model that predicts accident severity (Slight, Serious, Fatal) based on input parameters
User-friendly GUI: Built with CustomTkinter for an enhanced user experience
Data Analysis: Comprehensive exploration of accident data relationships

Dataset Information:
The system uses a CSV dataset (dataset.csv) with the following key columns:
Accident Date: Date of accident (format: DD/MM/YYYY)
Time (24hr): Time of accident in 24-hour format
Weather Conditions: Coded weather conditions (1-9)
Number of Vehicles: Count of vehicles involved
Number of Casualties: Count of casualties
Road Surface: Coded road surface conditions (1-6,9)
Lighting Conditions: Lighting conditions at time of accident
Accident_Severity: Target variable (1=Slight, 2=Serious, 3=Fatal)

Code Mappings:
Weather Conditions:
1: Clear/Fine
2: Raining
3: Snowing
4: Fog/Mist
5: Strong Wind
6: Storm
7: Hailstorm
8: Humid
9: Other

Road Surface Conditions:
1: Dry
2: Wet or Damp
3: Snow
4: Frost/Ice
6: Oil/Spillage
9: Flood

Accident Severity:
1: Slight
2: Serious
3: Fatal

Installation Requirements:
Prerequisites
Python 3.7 or higher
pip (Python package manager)

Dependencies:
Install the required packages using:
pip install pandas matplotlib seaborn scikit-learn customtkinter numpy

Usage
Ensure your dataset is named dataset.csv and placed in the same directory as the script

Run the application:
python perfect.py

Use the interface to:

Visualize accident data by different factors

Predict accident severity by providing input parameters

Input Parameters for Prediction
When using the prediction feature, provide the following inputs:

Time (24hr): Hour of day (0-23)

Weather Conditions: Numeric code (1-9) as defined above

Number of Vehicles: Integer count

Number of Casualties: Integer count

Road Surface: Numeric code (1,2,3,4,6,9) as defined above

Lighting Conditions: Numeric code (1=Day, 2=Night, 3=Dark)

Project Structure
accident-analysis-system/
├── perfect.py # Main application code
├── dataset.csv # Accident dataset (not included in repo)
├── requirements.txt # Python dependencies
└── README.md # Project documentation

Machine Learning Model
The system uses a Random Forest Classifier with:

100 estimators

80/20 train-test split

Random state for reproducibility

The model predicts accident severity (1, 2, or 3) and provides a confidence percentage for the prediction.

Contributing
Fork the repository

Create a feature branch

Make your changes

Test thoroughly

Submit a pull request

License
This project is open source and available under the MIT License.

Acknowledgments
Built with Python data science libraries (pandas, scikit-learn, matplotlib, seaborn)

GUI implemented with CustomTkinter

Machine learning model: Random Forest Classifier

Support
For questions or issues regarding this project, please open an issue in the GitHub repository.

