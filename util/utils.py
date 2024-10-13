import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from config import *

def get_categories(df, col, default_value):
    unique_values = list(sorted(df[col].unique()))
    # default value will work as Total
    unique_values = [x for x in unique_values if x != 'Total']
    
    return [default_value] + unique_values

def merge(left:pd.DataFrame, right:pd.DataFrame, key=None, how='inner'):
    common = set(left.columns) & set(right.columns)
    
    # if no key is provided, assume the common columns that 
    # ends with '_ID' or  '_YEAR' are the common keys
    if key is None:
        key = [
            column for column in common 
                if left[column].dtype in [object, str] or column.endswith('_YEAR') or column.endswith('_ID')
        ]
        if len(key) == 0: return None
    
    if type(key) != list: 
        common = [col for col in common if col != key]
    else:
        common = [col for col in common if col not in key]
    
    merged_data = left.merge(right.drop(columns=common), on = key, how=how)
    # logger.info(f'Shape: left {left.shape}, right {right.shape}, merged {merged_data.shape}.\n')
    
    return merged_data

def predict(summed, predictions):
    years = summed[time_column].values
    
    for time_step in range(seq_len, len(years)):
        Y = summed[summed[time_column]<years[time_step]][target].values
        
        model = ARIMA(Y, order=(seq_len, 1, 1))
        model.initialize_approximate_diffuse()
        model_fitted = model.fit() 
        
        result = model_fitted.get_forecast()
        forecast = result.predicted_mean
        forecast = np.round(np.where(forecast<=0, 0, forecast), 0)
        
        predictions['PREDICTED_MEAN'].append(forecast[0])
        predictions[time_column].append(years[time_step])
        
        ci = result.conf_int(alpha=alpha)
        ci = np.round(np.where(ci<=0, 0, ci), 0)
        predictions[f'{confidence}% CI - LOWER'].append(ci[0,0])
        predictions[f'{confidence}% CI - UPPER'].append(ci[0,1])
    
    return predictions

def autoregressive(summed, predictions):
    Y = list(summed[target].values)
    if len(Y) == 0:
        return predictions

    for i in range(0, pred_len):
        model = ARIMA(Y, order=(seq_len, 1, 1))
        model.initialize_approximate_diffuse()
        model_fitted = model.fit() 
        result = model_fitted.get_forecast()
        forecast = result.predicted_mean
        forecast = np.round(np.where(forecast<=0, 0, forecast), 0)
        
        predictions['PREDICTED_MEAN'].append(forecast[0])
        predictions[time_column].append(summed[time_column].values[-1]+i+1)
        
        ci = result.conf_int(alpha=alpha)
        ci = np.round(np.where(ci<=0, 0, ci), 0)
        predictions[f'{confidence}% CI - LOWER'].append(ci[0,0])
        predictions[f'{confidence}% CI - UPPER'].append(ci[0,1])
        Y.append(forecast[0])
        
    return predictions


def filter_disbursement(
    df, level, program_desc, academic_plan, 
    report_category, report_code, need_based, 
    residency, logger
):
    dis = df.copy()
    logger.info(f'Initial size: {len(dis)}')
        
    for (var, column) in [
        (program_desc,'ACADEMIC_PROGRAM_DESC'),
        (level, 'ACADEMIC_LEVEL_TERM_START'),
        (academic_plan, 'ACADEMIC_PLAN'),
        (report_category, 'UVA_ACCESS'),
        (report_code, 'REPORT_CODE'),
        (need_based, 'Need based'),
        (residency, 'FIN_AID_FED_RES')
    ]:
        
        logger.info(f'Filtering {column} == {var}...')
        # logger.info(f'Unique values: {list(dis[column].unique())}')
        default_value = default_values[column]
        
        if var == default_value or var == 'Total':
            if 'Total' in dis[column].values:  
                dis = dis[dis[column] == 'Total']  
            elif dis[column].nunique() != 1:
                logger.error(f'{column} has {dis[column].nunique()} unique values but does not contain Total.')   
                
        else: 
            if var not in dis[column].values: 
                logger.error(f'Error ! {column} does not contain {var}.')
                logger.info(f'Unique values: {list(dis[column].unique())}')
            dis = dis[dis[column] == var]
            
        if len(dis) == 0:
            if var == default_value: 
                logger.error(f'{column} does not contain Total. Please check data.')
            else:
                logger.error(f'{column} does not contain {var}. Please check data.')
            break
    
        logger.info(f'Reduced size: {len(dis)}.')
    
    logger.info('Done\n')
    return dis