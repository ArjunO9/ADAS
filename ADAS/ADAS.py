# Copyright (c) 2024 Arjun Soni
# All Rights Reserved. This software is the proprietary property of Arjun Soni.
# Unauthorized use, reproduction, or distribution is strictly prohibited.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np


# Load dataset
df = pd.read_csv('dataset.csv')

# Data preprocessing (optional)
df['Accident Date'] = pd.to_datetime(df['Accident Date'], format='%d/%m/%Y')

# Mapping legends for user input and visualization
weather_legend = {
    1: 'Clear/Fine',
    2: 'Raining',
    3: 'Snowing',
    4: 'Fog/Mist',
    5: 'Strong Wind',
    6: 'Storm',
    7: 'hailstorm',
    8: 'humid',
    9: 'other'
}

road_surface_legend = {
    1: 'Dry',
    2: 'Wet or Damp',
    3: 'Snow',
    4: 'Frost/Ice',
    9: 'Flood',
    6: 'Oil/Spillage'
}

severity_legend = {
    1: 'Slight',
    2: 'Serious',
    3: 'Fatal'
}

# Select relevant features for training
X = df[['Time (24hr)', 'Weather Conditions', 'Number of Vehicles', 
        'Number of Casualties', 'Road Surface', 'Lighting Conditions']]
y = df['Accident_Severity']  # Target variable (Accident Severity)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function to clear the display area
def clear_display():
    for widget in display_frame.winfo_children():
        widget.destroy()

# Function to create visualizations
def plot_visualizations(attribute):
    clear_display()  # Clear previous visualizations
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if attribute == 'Weather':
        plot = sns.countplot(x='Weather Conditions', hue='Accident_Severity', data=df, palette='coolwarm', ax=ax)
        ax.set_title('Accident Count by Weather Conditions and Severity')
        ax.set_xlabel('Weather Conditions')
        ax.set_ylabel('Accident Count')

        # Add exact counts above bars
        for p in plot.patches:
            plot.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                          ha='center', va='center', fontsize=11, color='black', xytext=(0, 8), textcoords='offset points')

        # Add legend for weather conditions
        labels = [weather_legend.get(int(tick.get_text()), tick.get_text()) for tick in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        ax.legend(title='Accident Severity', labels=[severity_legend.get(h, h) for h in sorted(df['Accident_Severity'].unique())])

    elif attribute == 'Time':
        plot = sns.histplot(df, x='Time (24hr)', hue='Accident_Severity', kde=False, bins=24, ax=ax, palette='coolwarm')
        ax.set_title('Accident Distribution by Time (24hr) and Severity')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Accident Count')

        # Add exact counts above bars
        for patch in plot.patches:
            ax.annotate(f'{int(patch.get_height())}', (patch.get_x() + patch.get_width() / 2., patch.get_height()),
                        ha='center', va='center', fontsize=11, color='black', xytext=(0, 8), textcoords='offset points')
        
        ax.legend(title='Accident Severity', labels=[severity_legend.get(h, h) for h in sorted(df['Accident_Severity'].unique())])

    elif attribute == 'Vehicles':
        plot = sns.countplot(x='Number of Vehicles', hue='Accident_Severity', data=df, palette='viridis', ax=ax)
        ax.set_title('Accident Count by Number of Vehicles and Severity')
        ax.set_xlabel('Number of Vehicles')
        ax.set_ylabel('Accident Count')

        # Add exact counts above bars
        for p in plot.patches:
            plot.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                          ha='center', va='center', fontsize=11, color='black', xytext=(0, 8), textcoords='offset points')
        
        ax.legend(title='Accident Severity', labels=[severity_legend.get(h, h) for h in sorted(df['Accident_Severity'].unique())])

    elif attribute == 'Road Surface':
        plot = sns.countplot(x='Road Surface', hue='Accident_Severity', data=df, palette='magma', ax=ax)
        ax.set_title('Accident Count by Road Surface Conditions and Severity')
        ax.set_xlabel('Road Surface Condition')
        ax.set_ylabel('Accident Count')

        # Add exact counts above bars
        for p in plot.patches:
            plot.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                          ha='center', va='center', fontsize=11, color='black', xytext=(0, 8), textcoords='offset points')
        
        # Add legend for road surface conditions
        labels = [road_surface_legend.get(int(tick.get_text()), tick.get_text()) for tick in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        ax.legend(title='Accident Severity', labels=[severity_legend.get(h, h) for h in sorted(df['Accident_Severity'].unique())])

    # Show the plot in the GUI
    canvas = FigureCanvasTkAgg(fig, master=display_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Function to predict accident severity with input legends
def predict_severity():
    clear_display()  # Clear previous prediction display
    
    def get_user_input():
        try:
            time_of_day = int(time_entry.get())
            weather = int(weather_entry.get())
            vehicles = int(vehicles_entry.get())
            casualties = int(casualties_entry.get())
            road_surface = int(road_surface_entry.get())
            lighting = int(lighting_entry.get())

            # Prepare input as a dataframe
            input_data = pd.DataFrame({
                'Time (24hr)': [time_of_day],
                'Weather Conditions': [weather],
                'Number of Vehicles': [vehicles],
                'Number of Casualties': [casualties],
                'Road Surface': [road_surface],
                'Lighting Conditions': [lighting]
            })

            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            highest_proba = np.max(prediction_proba) * 100  # Get the highest probability

            # Display prediction result in percentage
            result_label = ctk.CTkLabel(display_frame, text=f"Predicted Accident Severity: {severity_legend[prediction]} ({highest_proba:.2f}% chance)", font=('Arial', 14))
            result_label.pack(pady=10)

            # Print user input along with the prediction
            user_input_label = ctk.CTkLabel(display_frame, text=f"User Input: Time={time_of_day}, Weather={weather}, Vehicles={vehicles}, Casualties={casualties}, Road Surface={road_surface}, Lighting={lighting}", font=('Arial', 12))
            user_input_label.pack(pady=10)

        except IndexError as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction: {str(e)}")
        except ValueError as e:
            messagebox.showerror("Input Error", f"Please provide valid inputs: {str(e)}")

    # Create form to take user input for prediction
    input_frame = ctk.CTkFrame(display_frame)
    input_frame.pack(pady=10)

    ctk.CTkLabel(input_frame, text="Time (24hr):").grid(row=0, column=0, padx=5, pady=5)
    time_entry = ctk.CTkEntry(input_frame)
    time_entry.grid(row=0, column=1, padx=5, pady=5)

    ctk.CTkLabel(input_frame, text="Weather Conditions (1:Clear, 2:Raining, 3:Snow, 4:Fog, 5:Wind, 6:Other):").grid(row=1, column=0, padx=5, pady=5)
    weather_entry = ctk.CTkEntry(input_frame)
    weather_entry.grid(row=1, column=1, padx=5, pady=5)

    ctk.CTkLabel(input_frame, text="Number of Vehicles:").grid(row=2, column=0, padx=5, pady=5)
    vehicles_entry = ctk.CTkEntry(input_frame)
    vehicles_entry.grid(row=2, column=1, padx=5, pady=5)

    ctk.CTkLabel(input_frame, text="Number of Casualties:").grid(row=3, column=0, padx=5, pady=5)
    casualties_entry = ctk.CTkEntry(input_frame)
    casualties_entry.grid(row=3, column=1, padx=5, pady=5)

    ctk.CTkLabel(input_frame, text="Road Surface (1:Dry, 2:Wet, 3:Snow, 4:Frost, 5:Flood, 6:Oil):").grid(row=4, column=0, padx=5, pady=5)
    road_surface_entry = ctk.CTkEntry(input_frame)
    road_surface_entry.grid(row=4, column=1, padx=5, pady=5)

    ctk.CTkLabel(input_frame, text="Lighting Conditions (1:Day, 2:Night, 3:Dark):").grid(row=5, column=0, padx=5, pady=5)
    lighting_entry = ctk.CTkEntry(input_frame)
    lighting_entry.grid(row=5, column=1, padx=5, pady=5)

    submit_button = ctk.CTkButton(input_frame, text="Submit", command=get_user_input)
    submit_button.grid(row=6, columnspan=2, pady=10)

# Set up the main application window
app = ctk.CTk()
app.title("Accident Data Analysis System")

# Create main frames
input_frame = ctk.CTkFrame(app)
input_frame.pack(pady=20)

display_frame = ctk.CTkFrame(app)
display_frame.pack(pady=20)

# Create buttons for functionalities
visualize_button = ctk.CTkButton(input_frame, text="Visualize Weather Data", command=lambda: plot_visualizations('Weather'))
visualize_button.grid(row=0, column=0, padx=10, pady=10)

visualize_time_button = ctk.CTkButton(input_frame, text="Visualize Time Data", command=lambda: plot_visualizations('Time'))
visualize_time_button.grid(row=0, column=1, padx=10, pady=10)

visualize_vehicles_button = ctk.CTkButton(input_frame, text="Visualize Vehicles Data", command=lambda: plot_visualizations('Vehicles'))
visualize_vehicles_button.grid(row=0, column=2, padx=10, pady=10)

visualize_road_button = ctk.CTkButton(input_frame, text="Visualize Road Surface Data", command=lambda: plot_visualizations('Road Surface'))
visualize_road_button.grid(row=0, column=3, padx=10, pady=10)

predict_button = ctk.CTkButton(input_frame, text="Predict Accident Severity", command=predict_severity)
predict_button.grid(row=1, columnspan=4, pady=10)

# Run the application
app.mainloop()