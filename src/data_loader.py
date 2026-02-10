import pandas as pd
import os

def get_column_names():
    """Returns the standard column names for the NASA C-MAPSS dataset."""
    cols = ['unit_id', 'time', 'setting1', 'setting2', 'setting3']
    cols += [f's{i}' for i in range(1, 22)]
    return cols

def validate_data(df, file_name):
    """
    Performs basic data integrity checks.
    
    Args:
        df (pd.DataFrame): The loaded dataframe.
        file_name (str): Name of the file for error reporting.
    """
    # 1. Check for Nulls
    if df.isnull().values.any():
        null_count = df.isnull().sum().sum()
        print(f"   ⚠️ WARNING: {file_name} contains {null_count} missing values.")
    
    # 2. Check Column Count (Expect 26 for Train/Test)
    expected_cols = 26
    if "RUL" not in file_name and len(df.columns) != expected_cols:
        print(f"   ❌ ERROR: {file_name} has {len(df.columns)} columns. Expected {expected_cols}.")
    
    # 3. Check for Negative Time/Unit IDs
    if (df['unit_id'] < 1).any() or (df['time'] < 1).any():
         print(f"   ⚠️ WARNING: {file_name} contains non-positive IDs or time cycles.")

def read_raw_txt(file_path):
    """Reads a single raw .txt file and validates it."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw file not found: {file_path}")
    
    col_names = get_column_names()
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    
    # Validate before returning
    validate_data(df, os.path.basename(file_path))
    return df

def save_interim_csv(df, output_path):
    """Saves the dataframe to the interim folder."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"   Converted and Validated: {os.path.basename(output_path)}")

def convert_all_raw_data(raw_dir="data/raw", interim_dir="data/interim"):
    """Main orchestration function for conversion and validation."""
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    print(f"🚀 Initializing Data Loader...")
    
    for ds in datasets:
        # Convert Train/Test
        for mode in ['train', 'test']:
            src = os.path.join(raw_dir, f"{mode}_{ds}.txt")
            dst = os.path.join(interim_dir, f"{mode}_{ds}.csv")
            if os.path.exists(src):
                df = read_raw_txt(src)
                save_interim_csv(df, dst)
            
        # Convert RUL (Ground Truth)
        rul_src = os.path.join(raw_dir, f"RUL_{ds}.txt")
        rul_dst = os.path.join(interim_dir, f"RUL_{ds}.csv")
        if os.path.exists(rul_src):
            df_rul = pd.read_csv(rul_src, header=None, names=['RUL'])
            save_interim_csv(df_rul, rul_dst)

if __name__ == "__main__":
    convert_all_raw_data()