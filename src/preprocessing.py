import pandas as pd

def preprocess_data(df):
    """
    Direct preprocessing pipeline:
    1. Convert weight and height to Metric (kg/cm) (in original data set it wasnt)
    2. Filter out biologically impossible records (bf <3% and height <120)
    3. Remove non-predictive columns (density)
    """
    df = df.copy()

     
    # Weight: lbs -> kg
    # Height: inch -> cm
    if 'Weight' in df.columns:
        df['Weight'] = (df['Weight'] * 0.453592).round(2)
        
    if 'Height' in df.columns:
        df['Height'] = (df['Height'] * 2.54).round(2)

    
    # Delete rows where Body Fat is less than 3%
    if 'class' in df.columns:
        df = df[df['class'] >= 3.0]
        
    # Delete rows where Height is less than 120 cm
    if 'Height' in df.columns:
        df = df[df['Height'] >= 120.0]

    
    # Remove Density (leaks the answer)
    if 'Density' in df.columns:
        df = df.drop(columns=['Density'])
    
    return df