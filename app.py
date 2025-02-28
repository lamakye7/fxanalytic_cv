import gradio as gr
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

class ForexChartAnalyzer:
    def __init__(self):
        # Initialize models
        self.model1 = YOLO('best (11).pt')  # Replace with your model paths
        self.model2 = YOLO('best (12).pt')

        # Path to your CSV file in Google Drive
        self.csv_path = 'forex_predictions.csv'

        # Load existing predictions or create new DataFrame
        if os.path.exists(self.csv_path):
            self.predictions_df = pd.read_csv(self.csv_path)
        else:
            self.predictions_df = pd.DataFrame(columns=[
                'timestamp', 'model_used', 'class_name', 'confidence',
                'fib_level', 'timeframe', 'forex_symbol', 'conf_threshold',
                'filename', 'image_path'
            ])

    def save_to_drive(self):
        """Save prediction"""
        self.predictions_df.to_csv(self.csv_path, index=False)

    def predict_and_visualize(self, image, image_path, model_choice, conf_threshold,
                            fib_level, timeframe, forex_symbol):
        # Extract filename from path
        filename = os.path.basename(image_path) if image_path else "uploaded_image"

        # Select model based on choice
        model = self.model1 if model_choice == "Model 1" else self.model2

        # Make prediction
        results = model(image, conf=conf_threshold)[0]

        # Process predictions
        current_predictions = []
        timestamp = datetime.now()

        # Check if any detections were made
        if len(results.boxes) > 0:
            for box in results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = results.names[class_id]

                current_predictions.append({
                    'timestamp': timestamp,
                    'model_used': model_choice,
                    'class_name': class_name,
                    'confidence': confidence,
                    'fib_level': fib_level,
                    'timeframe': timeframe,
                    'forex_symbol': forex_symbol,
                    'conf_threshold': conf_threshold,
                    'filename': filename,
                    'image_path': image_path
                })
        else:
            # Add a row indicating no detection
            current_predictions.append({
                'timestamp': timestamp,
                'model_used': model_choice,
                'class_name': 'no_detection',
                'confidence': 0.0,
                'fib_level': fib_level,
                'timeframe': timeframe,
                'forex_symbol': forex_symbol,
                'conf_threshold': conf_threshold,
                'filename': filename,
                'image_path': image_path
            })

        # Add new predictions to DataFrame
        new_predictions = pd.DataFrame(current_predictions)
        self.predictions_df = pd.concat([self.predictions_df, new_predictions], ignore_index=True)

        # Save updated DataFrame to Google Drive
        self.save_to_drive()

        # Create visualizations
        annotated_image = results.plot()

        # Create metrics visualization
        class_counts = self.predictions_df['class_name'].value_counts()
        fig_metrics = px.bar(
            x=class_counts.index,
            y=class_counts.values,
            title='Detection Class Distribution',
            labels={'x': 'Class', 'y': 'Count'}
        )

        # Create trading metrics
        trading_metrics = self.calculate_trading_metrics()

        return annotated_image, fig_metrics, trading_metrics

    def calculate_trading_metrics(self):
        if len(self.predictions_df) == 0:
            return "No predictions yet"

        # Calculate basic trading metrics
        total_predictions = len(self.predictions_df)
        unique_patterns = self.predictions_df['class_name'].nunique()
        #recent_patterns = self.predictions_df.tail(5)['class_name'].tolist()

        metrics_text = f"""
        Trading Metrics:
        - Total Analyses: {total_predictions}
        - Unique Patterns Detected: {unique_patterns}
        """
        return metrics_text

# Create Gradio Interface
analyzer = ForexChartAnalyzer()

def analyze_chart(image, model_choice, conf_threshold, fib_level, timeframe, forex_symbol):
    # Get the temporary path of the uploaded image
    image_path = image.name if hasattr(image, 'name') else None
    return analyzer.predict_and_visualize(
        image, image_path, model_choice, conf_threshold, fib_level, timeframe, forex_symbol
    )

# Define choices for dropdowns
timeframe_choices = ["1m", "5m", "15m", "30m", "45m", "1h", "4h", "1d", "1w", "1M" ]
forex_pairs = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
    "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP",
    "EUR/JPY", "GBP/JPY", "XAU/USD"
]

# Create the interface
iface = gr.Interface(
    fn=analyze_chart,
    inputs=[
        gr.Image(type="numpy", label="Upload Chart Image"),
        gr.Radio(["Model 1", "Model 2"], label="Select Model", value="Model 1"),
        gr.Slider(minimum=0.25, maximum=1.0, value=0.45, label="Confidence Threshold"),
        gr.Number(label="Fibonacci Level", value=0.618),
        gr.Dropdown(choices=timeframe_choices, label="Timeframe", value="1h"),
        gr.Dropdown(choices=forex_pairs, label="Forex Symbol", value="EUR/USD")
    ],
    outputs=[
        gr.Image(label="Annotated Chart"),
        gr.Plot(label="Detection Distribution"),
        gr.Textbox(label="Trading Metrics")
    ],
    title="Forex Chart Pattern Analyzer",
    description="Upload a forex chart to detect and analyze patterns using custom YOLO models"
)

if __name__ == "__main__":
    iface.launch()