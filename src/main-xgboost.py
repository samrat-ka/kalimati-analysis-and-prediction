import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
from tqdm import tqdm
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import joblib

warnings.filterwarnings('ignore')

class XGBoostVegetablePriceForecaster:
    def __init__(self, data_file_path, lookback_days=30):
        """
        Initialize XGBoost forecaster
        
        Args:
            data_file_path (str): Path to CSV file
            lookback_days (int): Number of previous days to use as features
        """
        self.data_file_path = data_file_path
        self.lookback_days = lookback_days
        self.data = None
        self.commodities = []
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.predictions = {}
        self.metrics = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data"""
        print("Loading data...")
        self.data = pd.read_csv(self.data_file_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.commodities = self.data['Commodity'].unique()
        
        print(f"Data loaded successfully!")
        print(f"Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        print(f"Number of commodities: {len(self.commodities)}")
        print(f"Total records: {len(self.data)}")
        
        return self.data
    
    def create_time_features(self, df):
        """
        Create time-based features from date
        
        Args:
            df (DataFrame): Input dataframe with Date column
            
        Returns:
            DataFrame: Dataframe with additional time features
        """
        df = df.copy()
        

        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['Quarter'] = df['Date'].dt.quarter
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        

        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        

        df['IsFestivalSeason'] = ((df['Month'] == 10) | (df['Month'] == 11) | 
                                 (df['Month'] == 3) | (df['Month'] == 4)).astype(int)
        

        df['IsMonsoon'] = ((df['Month'] >= 6) & (df['Month'] <= 9)).astype(int)
        
        return df
    
    def create_lag_features(self, df, target_col='Average'):
        """
        Create lag features for time series
        
        Args:
            df (DataFrame): Input dataframe
            target_col (str): Target column to create lags for
            
        Returns:
            DataFrame: Dataframe with lag features
        """
        df = df.copy()
        

        lag_periods = [1, 2, 3, 5, 7, 14, 21, 30]
        
        for lag in lag_periods:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        

        windows = [7, 14, 30]
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
        

        df[f'{target_col}_pct_change_1'] = df[target_col].pct_change(1)
        df[f'{target_col}_pct_change_7'] = df[target_col].pct_change(7)
        df[f'{target_col}_pct_change_30'] = df[target_col].pct_change(30)

        df[f'{target_col}_volatility_7'] = df[target_col].rolling(7).std() / df[target_col].rolling(7).mean()
        df[f'{target_col}_volatility_30'] = df[target_col].rolling(30).std() / df[target_col].rolling(30).mean()
        
        return df
    
    def prepare_commodity_data(self, commodity):
        """
        Prepare data for a specific commodity with all features
        
        Args:
            commodity (str): Commodity name
            
        Returns:
            DataFrame: Prepared data with features
        """

        commodity_data = self.data[self.data['Commodity'] == commodity].copy()
        commodity_data = commodity_data.sort_values('Date').reset_index(drop=True)
        

        commodity_data = self.create_time_features(commodity_data)
        

        commodity_data = self.create_lag_features(commodity_data, 'Average')
        commodity_data = self.create_lag_features(commodity_data, 'Minimum')
        commodity_data = self.create_lag_features(commodity_data, 'Maximum')

        commodity_data['Price_Spread'] = commodity_data['Maximum'] - commodity_data['Minimum']
        commodity_data['Price_Spread_Pct'] = commodity_data['Price_Spread'] / commodity_data['Average']
        

        commodity_data = commodity_data.dropna().reset_index(drop=True)
        
        return commodity_data
    
    def get_feature_columns(self, df):
        """Get list of feature columns (excluding Date, Commodity, and target)"""
        exclude_cols = ['Date', 'Commodity', 'Average', 'Minimum', 'Maximum', 'Unit']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def train_commodity_model(self, commodity):
        """
        Train XGBoost model for a specific commodity
        
        Args:
            commodity(str): Commodity name
            
        Returns:
            dict: Training results
        """

        commodity_data = self.prepare_commodity_data(commodity)
        

        if len(commodity_data) < 100:
            return None
        

        feature_cols = self.get_feature_columns(commodity_data)
        X = commodity_data[feature_cols]
        y = commodity_data['Average']
        

        X = X.fillna(X.median())
        

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.scalers[commodity] = scaler
        

        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        

        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        

        model.fit(X_train, y_train)
        

        y_pred = model.predict(X_test)
        

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        

        self.models[commodity] = model
        self.metrics[commodity] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        return {
            'model': model,
            'metrics': self.metrics[commodity],
            'feature_cols': feature_cols,
            'test_dates': commodity_data['Date'].iloc[split_idx:].values,
            'y_test_actual': y_test.values,
            'y_pred': y_pred
        }
    
    def forecast_commodity(self, commodity, days=30):
        """
        Generate future predictions for a commodity
        
        Args:
            commodity (str): Commodity name
            days (int): Number of days to forecast
            
        Returns:
            DataFrame: Predictions
        """
        if commodity not in self.models:
            print(f"No trained model found for {commodity}")
            return None
        
        model = self.models[commodity]
        scaler = self.scalers[commodity]
        

        commodity_data = self.prepare_commodity_data(commodity)
        
        if commodity_data.empty:
            return None
        

        feature_cols = self.get_feature_columns(commodity_data)
        
        predictions = []
        last_date = commodity_data['Date'].max()
        

        current_data = commodity_data.tail(self.lookback_days * 2).copy()
        
        for i in range(days):
            try:

                latest_row = current_data.iloc[-1:][feature_cols]
                latest_row = latest_row.fillna(latest_row.mean())

                latest_scaled = scaler.transform(latest_row)

                pred_price = model.predict(latest_scaled)[0]
                

                pred_date = last_date + timedelta(days=i+1)
                predictions.append({
                    'Date': pred_date,
                    'Predicted_Price': pred_price,
                    'Commodity': commodity
                })
                

                next_row = current_data.iloc[-1:].copy()
                next_row['Date'] = pred_date
                next_row['Average'] = pred_price
                next_row['Minimum'] = pred_price * 0.95 
                next_row['Maximum'] = pred_price * 1.05
                

                next_row = self.create_time_features(next_row)

                current_data = pd.concat([current_data, next_row], ignore_index=True)

                current_data = self.create_lag_features(current_data, 'Average')
                current_data = self.create_lag_features(current_data, 'Minimum') 
                current_data = self.create_lag_features(current_data, 'Maximum')

                current_data['Price_Spread'] = current_data['Maximum'] - current_data['Minimum']
                current_data['Price_Spread_Pct'] = current_data['Price_Spread'] / current_data['Average']

                current_data = current_data.tail(self.lookback_days * 3)
                
            except Exception as e:
                print(f"Error in prediction step {i+1} for {commodity}: {str(e)}")
                break
        
        return pd.DataFrame(predictions)
    
    def train_all_models(self):
        """Train XGBoost models for all commodities"""
        print("\nTraining XGBoost models for all commodities...")
        
        successful_models = 0
        
        for commodity in tqdm(self.commodities, desc="Training XGBoost models"):
            try:
                result = self.train_commodity_model(commodity)
                if result is not None:
                    successful_models += 1
                else:
                    tqdm.write(f"  Skipped {commodity} - insufficient data")
            except Exception as e:
                tqdm.write(f"  Error training {commodity}: {str(e)}")
                continue
        
        print(f"\nSuccessfully trained {successful_models} models out of {len(self.commodities)} commodities")
    
    def generate_all_forecasts(self, days=30):
        """Generate forecasts for all trained models"""
        print(f"\nGenerating {days}-day forecasts...")
        
        all_predictions = []
        
        for commodity in tqdm(self.models.keys(), desc="Generating forecasts"):
            try:
                pred_df = self.forecast_commodity(commodity, days)
                if pred_df is not None:
                    all_predictions.append(pred_df)
            except Exception as e:
                tqdm.write(f"  Error forecasting {commodity}: {str(e)}")
                continue
        
        if all_predictions:
            self.all_predictions_df = pd.concat(all_predictions, ignore_index=True)
            print(f"Generated forecasts for {len(self.models)} commodities")
        else:
            self.all_predictions_df = pd.DataFrame()
    
    def get_feature_importance(self, commodity, top_n=15):
        """
        Get feature importance for a commodity model
        
        Args:
            commodity (str): Commodity name
            top_n (int): Number of top features to return
            
        Returns:
            DataFrame: Feature importance
        """
        if commodity not in self.models:
            return None
        
        model = self.models[commodity]
        commodity_data = self.prepare_commodity_data(commodity)
        feature_cols = self.get_feature_columns(commodity_data)

        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(top_n)
        
        return feature_importance
    
    def create_feature_importance_plots(self, output_dir='xgboost_plots', top_commodities=5):
        """Create feature importance plots"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\nCreating feature importance plots...")

        commodity_counts = self.data['Commodity'].value_counts()
        top_commodities_list = commodity_counts.head(top_commodities).index.tolist()
        
        for commodity in top_commodities_list:
            if commodity not in self.models:
                continue
                
            importance_df = self.get_feature_importance(commodity, 15)
            
            if importance_df is not None:
                plt.figure(figsize=(12, 8))
                plt.barh(range(len(importance_df)), importance_df['Importance'])
                plt.yticks(range(len(importance_df)), importance_df['Feature'])
                plt.xlabel('Feature Importance')
                plt.title(f'Top 15 Feature Importance - {commodity}')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                
                safe_name = commodity.replace('/', '_').replace('(', '').replace(')', '').replace(' ', '_')
                plt.savefig(f'{output_dir}/{safe_name}_feature_importance.png', 
                          dpi=300, bbox_inches='tight')
                plt.close()
    
    def create_performance_plots(self, output_dir='xgboost_plots'):
        """Create performance visualization plots"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\nCreating XGBoost performance plots...")
        
        if not self.metrics:
            return

        plt.figure(figsize=(16, 12))

        commodities = list(self.metrics.keys())
        mae_scores = [self.metrics[c]['MAE'] for c in commodities]
        rmse_scores = [self.metrics[c]['RMSE'] for c in commodities]
        r2_scores = [self.metrics[c]['R2'] for c in commodities]

        plt.subplot(2, 3, 1)
        plt.bar(range(len(commodities)), mae_scores, color='skyblue')
        plt.title('Mean Absolute Error by Commodity')
        plt.ylabel('MAE')
        plt.xticks(range(len(commodities)), [c[:10] + '...' if len(c) > 10 else c 
                  for c in commodities], rotation=45)

        plt.subplot(2, 3, 2)
        plt.bar(range(len(commodities)), rmse_scores, color='lightcoral')
        plt.title('Root Mean Square Error by Commodity')
        plt.ylabel('RMSE')
        plt.xticks(range(len(commodities)), [c[:10] + '...' if len(c) > 10 else c 
                  for c in commodities], rotation=45)

        plt.subplot(2, 3, 3)
        plt.bar(range(len(commodities)), r2_scores, color='lightgreen')
        plt.title('R² Score by Commodity')
        plt.ylabel('R²')
        plt.xticks(range(len(commodities)), [c[:10] + '...' if len(c) > 10 else c 
                  for c in commodities], rotation=45)

        plt.subplot(2, 3, 4)
        plt.hist(mae_scores, bins=20, alpha=0.7, color='blue')
        plt.title('Distribution of MAE Scores')
        plt.xlabel('MAE')
        
        plt.subplot(2, 3, 5)
        plt.hist(rmse_scores, bins=20, alpha=0.7, color='red')
        plt.title('Distribution of RMSE Scores')
        plt.xlabel('RMSE')
        
        plt.subplot(2, 3, 6)
        plt.hist(r2_scores, bins=20, alpha=0.7, color='green')
        plt.title('Distribution of R² Scores')
        plt.xlabel('R²')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/xgboost_performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_forecast_plots(self, output_dir='xgboost_plots'):
        """Create forecast visualization plots"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\nCreating forecast plots for top 10 commodities...")
        
        # Select top commodities by data availability
        commodity_counts = self.data['Commodity'].value_counts()
        top_commodities = commodity_counts.index.tolist()
        
        for commodity in tqdm(top_commodities, desc="Creating forecast plots"):
            if commodity not in self.models:
                continue
                
            try:

                commodity_data = self.data[self.data['Commodity'] == commodity].copy()
                commodity_data = commodity_data.sort_values('Date').reset_index(drop=True)
                

                recent_cutoff = commodity_data['Date'].max() - timedelta(days=180)
                recent_data = commodity_data[commodity_data['Date'] >= recent_cutoff]
                

                forecast_df = self.forecast_commodity(commodity, 30)
                
                if forecast_df is not None:

                    plt.figure(figsize=(14, 8))

                    plt.plot(recent_data['Date'], recent_data['Average'], 
                           'o-', color='blue', label='Historical Prices', linewidth=2, markersize=3)

                    plt.plot(forecast_df['Date'], forecast_df['Predicted_Price'], color='red', label='XGBoost Predictions', linewidth=2)
                    

                    plt.axvline(x=commodity_data['Date'].max(), color='green', 
                              linestyle='--', alpha=0.7, label='Forecast Start')
                    
                    plt.title(f'{commodity} - XGBoost Price Forecast (30 Days)', 
                             fontsize=14, fontweight='bold')
                    plt.xlabel('Date', fontsize=12)
                    plt.ylabel('Price', fontsize=12)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    safe_name = commodity.replace('/', '_').replace('(', '').replace(')', '').replace(' ', '_')
                    plt.savefig(f'{output_dir}/{safe_name}_xgboost_forecast.png', 
                              dpi=300, bbox_inches='tight')
                    plt.close()
                    
            except Exception as e:
                tqdm.write(f"Error plotting {commodity}: {str(e)}")
                continue
    
    def save_models(self, models_dir='xgboost_models'):
        """Save trained models and scalers"""
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        print(f"\nSaving models to {models_dir}/...")
        
        for commodity in self.models.keys():
            try:
                safe_name = commodity.replace('/', '_').replace('(', '').replace(')', '').replace(' ', '_')

                model_path = f"{models_dir}/{safe_name}_xgboost_model.pkl"
                joblib.dump(self.models[commodity], model_path)

                scaler_path = f"{models_dir}/{safe_name}_scaler.pkl"
                joblib.dump(self.scalers[commodity], scaler_path)
                
            except Exception as e:
                print(f"Error saving {commodity}: {str(e)}")
        
        print(f"Models saved successfully!")
    
    def save_predictions_to_csv(self, output_file='xgboost_predictions.csv'):
        """Save predictions to CSV"""
        if hasattr(self, 'all_predictions_df') and not self.all_predictions_df.empty:
            output_df = self.all_predictions_df.copy()
            output_df['Model'] = 'XGBoost'
            output_df['Forecast_Generated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            column_order = ['Date', 'Commodity', 'Predicted_Price', 'Model', 'Forecast_Generated']
            output_df = output_df[column_order]
            output_df = output_df.sort_values(['Commodity', 'Date']).reset_index(drop=True)
            
            output_df.to_csv(output_file, index=False)
            print(f"\nXGBoost predictions saved to: {output_file}")
            print(f"Total prediction records: {len(output_df)}")
        else:
            print("No predictions to save!")
    
    def generate_summary_report(self):
        """Generate summary report"""
        print("\n" + "="*60)
        print("XGBOOST VEGETABLE PRICE FORECASTING SUMMARY")
        print("="*60)
        
        if not self.metrics:
            print("No models trained!")
            return
        
        print(f"Total Models Trained: {len(self.models)}")
        print(f"Lookback Period: {self.lookback_days} days")

        avg_mae = np.mean([m['MAE'] for m in self.metrics.values()])
        avg_rmse = np.mean([m['RMSE'] for m in self.metrics.values()])
        avg_r2 = np.mean([m['R2'] for m in self.metrics.values()])
        
        print(f"\nAverage Model Performance:")
        print(f"  Mean Absolute Error: {avg_mae:.2f}")
        print(f"  Root Mean Square Error: {avg_rmse:.2f}")
        print(f"  R² Score: {avg_r2:.3f}")

        best_commodity = min(self.metrics.keys(), key=lambda x: self.metrics[x]['MAE'])
        worst_commodity = max(self.metrics.keys(), key=lambda x: self.metrics[x]['MAE'])
        
        print(f"\nBest Performing Model: {best_commodity}")
        print(f"  MAE: {self.metrics[best_commodity]['MAE']:.2f}")
        print(f"  RMSE: {self.metrics[best_commodity]['RMSE']:.2f}")
        print(f"  R2: {self.metrics[best_commodity]['R2']:.3f}")
        
        print(f"\nWorst Performing Model: {worst_commodity}")
        print(f"  MAE: {self.metrics[worst_commodity]['MAE']:.2f}")
        print(f"  RMSE: {self.metrics[worst_commodity]['RMSE']:.2f}")
        print(f"  R2: {self.metrics[worst_commodity]['R2']:.3f}")

        if best_commodity in self.models:
            print(f"\nTop 10 Features for Best Model ({best_commodity}):")
            importance_df = self.get_feature_importance(best_commodity, 10)
            for _, row in importance_df.iterrows():
                print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    def run_complete_analysis(self, predictions_file='xgboost_predictions.csv',
                            models_dir='xgboost_models', plots_dir='xgboost_plots'):
        """Run complete XGBoost analysis pipeline"""
        print("Starting XGBoost Vegetable Price Forecasting Pipeline...")
        print("=" * 60)
        

        self.load_and_prepare_data()
        

        self.train_all_models()

        self.generate_all_forecasts(30)

        self.save_predictions_to_csv(predictions_file)

        self.save_models(models_dir)

        self.create_performance_plots(plots_dir)
        self.create_feature_importance_plots(plots_dir)
        self.create_forecast_plots(plots_dir)

        self.generate_summary_report()
        
        print("\n" + "="*60)
        print("XGBOOST FORECASTING PIPELINE COMPLETED!")
        print("="*60)


def main():
    """Main function"""
    DATA_FILE = './fetcher/Kalimati/data/csv/data.csv'
    PREDICTIONS_OUTPUT = './data/xgboost_predictions.csv'
    MODELS_DIRECTORY = 'xgboost_models'
    PLOTS_DIRECTORY = 'xgboost_plots'
    
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found!")
        return
    
    # Initialize forecaster
    forecaster = XGBoostVegetablePriceForecaster(DATA_FILE, lookback_days=30)
    
    # Run complete analysis
    forecaster.run_complete_analysis(
        predictions_file=PREDICTIONS_OUTPUT,
        models_dir=MODELS_DIRECTORY,
        plots_dir=PLOTS_DIRECTORY
    )
    
    print(f"\nOutputs:")
    print(f"- XGBoost Predictions: {PREDICTIONS_OUTPUT}")
    print(f"- Saved Models: {MODELS_DIRECTORY}/ directory")
    print(f"- Plots: {PLOTS_DIRECTORY}/ directory")


if __name__ == "__main__":
    main()