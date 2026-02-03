import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from tqdm import tqdm

import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

import warnings
warnings.filterwarnings("ignore")

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class VegetablePriceForecaster:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.data = None
        self.commodities = []
        self.predictions = {}
        
    def load_and_prepare_data(self):
        print("Loading data...")
        self.data = pd.read_csv(self.data_file_path)
        
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        self.commodities = self.data['Commodity'].unique()
        
        print(f"Data loaded successfully!")
        print(f"Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        print(f"Number of commodities: {len(self.commodities)}")
        print(f"Total records: {len(self.data)}")
        
        return self.data
    
    def prepare_prophet_data(self, commodity_data):
        prophet_data = commodity_data[['Date', 'Average']].copy()
        prophet_data.columns = ['ds', 'y']
        
        # prophet_data = prophet_data.dropna()

        prophet_data = prophet_data.sort_values('ds').reset_index(drop=True)
        
        return prophet_data
    
    def create_prophet_model(self, commodity_data):
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        
        model.fit(commodity_data)
        return model
    
    def generate_predictions(self, model, periods_dict):
        predictions = {}
        
        for period_name, days in periods_dict.items():
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)

            predicted_values = forecast.tail(days)
            predictions[period_name] = predicted_values[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            
        return predictions
    
    def forecast_commodities(self):
        print("Starting forecasting for all commodities...")
        
        self.periods = {
            'next_1_day': 1,
            'next_7_days': 7,
            'next_15_days': 15,
            'next_30_days': 30
        }
        
        all_predictions = []
        
        for commodity in tqdm(self.commodities, desc="Forecasting commodities"):
            try:
                commodity_data = self.data[self.data['Commodity'] == commodity].copy()
                
                if len(commodity_data) < 30:
                    tqdm.write(f"  Skipping {commodity} - insufficient data ({len(commodity_data)} records)")
                    continue
                
                prophet_data = self.prepare_prophet_data(commodity_data)
                
                model = self.create_prophet_model(prophet_data)
                
                predictions = self.generate_predictions(model, self.periods)
                
                for period_name, pred_df in predictions.items():
                    pred_df_copy = pred_df.copy()
                    pred_df_copy['Commodity'] = commodity
                    pred_df_copy['Prediction_Period'] = period_name
                    pred_df_copy['Days_Ahead'] = self.periods[period_name]
                    
                    all_predictions.append(pred_df_copy)
                
                self.predictions[commodity] = {
                    'model': model,
                    'data': prophet_data,
                    'predictions': predictions
                }
                
            except Exception as e:
                tqdm.write(f"  Error processing {commodity}: {str(e)}")
                continue
        
        if all_predictions:
            self.all_predictions_df = pd.concat(all_predictions, ignore_index=True)
            print(f"Forecasting completed! Generated predictions for {len(self.predictions)} commodities.")
        else:
            print("No predictions generated!")
            self.all_predictions_df = pd.DataFrame()
    
    def save_predictions(self, output_file='price_predictions.csv'):
        
        if not self.predictions:
            print("No predictions to save!")
            return

        final_data = []
        for commodity, data in self.predictions.items():
            row_data = {'Commodity': commodity}
            for period_name, days in self.periods.items():
                pred_df = data['predictions'][period_name]
                if not pred_df.empty:
                    predicted_price = pred_df['yhat'].iloc[-1]
                    row_data[f'Predicted_Price_{days}days'] = predicted_price
                    row_data[f'Predicted_Selling_Price_{days}days'] = predicted_price * 1.20
            final_data.append(row_data)

        output_df = pd.DataFrame(final_data)
        
        if not output_df.empty:
            output_df = output_df.round(2)
            output_df.to_csv(output_file, index=False)
            print(f"\nPredictions saved to: {output_file}")
            print(f"Total prediction records: {len(output_df)}")
        else:
            print("No predictions generated to save!")
            
    def create_time_series_plots(self, output_dir='plots'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\nGenerating time series plots...")
        
        for commodity in tqdm(self.predictions.keys(), desc="Generating plots"):
            try:
                model = self.predictions[commodity]['model']
                historical_data = self.predictions[commodity]['data']
                
                last_date = historical_data['ds'].max()
                
                three_months_ago = last_date - timedelta(days=90)
                recent_data = historical_data[historical_data['ds'] >= three_months_ago].copy()
                
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)
                
                forecast_filtered = forecast[forecast['ds'] >= three_months_ago].copy()
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                historical_plot_data = recent_data.copy()
                ax.plot(historical_plot_data['ds'], historical_plot_data['y'], 
                       'o-', color='blue', label='Historical Prices', linewidth=2, markersize=3)
                
                ax.plot(forecast_filtered['ds'], forecast_filtered['yhat'], 
                       '-', color='orange', alpha=0.7, linewidth=1, label='Prophet Fit')
                
                prediction_data = forecast_filtered[forecast_filtered['ds'] > last_date]
                ax.plot(prediction_data['ds'], prediction_data['yhat'], 
                       'o-', color='red', label='30-Day Predictions', linewidth=2, markersize=4)
                

                # ax.fill_between(prediction_data['ds'], 
                #               prediction_data['yhat_lower'], 
                #               prediction_data['yhat_upper'], 
                #               color='red', alpha=0.2, label='Prediction Confidence')
                
                ax.axvline(x=last_date, color='green', linestyle='--', alpha=0.7)
                
                ax.set_title(f'{commodity} - Price Forecast\n(30 Days Prediction)', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Price', fontsize=12)
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                
                plt.xticks(rotation=45)
                plt.tight_layout()

                safe_commodity_name = commodity.replace('/', '_').replace('(', '').replace(')', '')
                plot_filename = f"{output_dir}/{safe_commodity_name}_forecast.png"
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                tqdm.write(f"  Error creating plot for {commodity}: {str(e)}")
                continue
        
        print(f"\nAll plots saved in '{output_dir}' directory")
    
    def generate_summary_report(self):
        if not hasattr(self, 'all_predictions_df') or self.all_predictions_df.empty:
            print("No predictions available for summary report!")
            return
        
        print("\n" + "="*60)
        print("VEGETABLE PRICE FORECASTING SUMMARY REPORT")
        print("="*60)
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        print(f"Report Generated: {current_date}")

        total_commodities = len(self.predictions)
        total_predictions = len(self.all_predictions_df)
        
        print(f"\nTotal Commodities Processed: {total_commodities}")
        print(f"Total Prediction Records: {total_predictions}")
        
        print("\nPrediction Period Breakdown:")
        period_counts = self.all_predictions_df['Prediction_Period'].value_counts()
        for period, count in period_counts.items():
            print(f"  {period}: {count} predictions")
        
        print("\n30-Day Price Predictions Summary:")
        thirty_day_preds = self.all_predictions_df[
            self.all_predictions_df['Prediction_Period'] == 'next_30_days'
        ]
        
        if not thirty_day_preds.empty:
            commodity_summary = thirty_day_preds.groupby('Commodity')['yhat'].agg([
                'min', 'max', 'mean'
            ]).round(2)
            
            print(f"{'Commodity':<25} {'Min Price':<10} {'Max Price':<10} {'Avg Price':<10}")
            print("-" * 60)
            for commodity, row in commodity_summary.iterrows():
                commodity_short = commodity[:22] + "..." if len(commodity) > 25 else commodity
                print(f"{commodity_short:<25} {row['min']:<10} {row['max']:<10} {row['mean']:<10}")
    
    def run_analysis(self, predictions_file='./data/predictions.csv', plots_dir='forecast_plots'):
        
        self.load_and_prepare_data()
        
        self.forecast_commodities()
        
        self.save_predictions(predictions_file)
        
        self.create_time_series_plots(plots_dir)
        
        self.generate_summary_report()
        
        print("\n" + "="*60)
        print("FORECASTING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)


def main():
    DATA_FILE = './fetcher/Kalimati/data/csv/data.csv'
    PREDICTIONS_OUTPUT = './data/predictions.csv'
    PLOTS_DIRECTORY = 'forecast_plots'


    forecaster = VegetablePriceForecaster(DATA_FILE)
    

    forecaster.run_analysis(
        predictions_file=PREDICTIONS_OUTPUT,
        plots_dir=PLOTS_DIRECTORY
    )
    
    print(f"\nOutputs:")
    print(f"- Predictions CSV: {PREDICTIONS_OUTPUT}")
    print(f"- Forecast plots: {PLOTS_DIRECTORY}/ directory")




if __name__ == "__main__":
    main()