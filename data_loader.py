import glob
import pandas as pd
import os

def load_all_stations(base_url="https://raw.githubusercontent.com/calantha168/Air_quality/main/data"):
    files = [
        "PRSA_Data_Aotizhongxin_20130301-20170228.csv",
        "PRSA_Data_Changping_20130301-20170228.csv",
        "PRSA_Data_Dingling_20130301-20170228.csv",
        "PRSA_Data_Dongsi_20130301-20170228.csv",
        "PRSA_Data_Guanyuan_20130301-20170228.csv",
        "PRSA_Data_Gucheng_20130301-20170228.csv",
        "PRSA_Data_Huairou_20130301-20170228.csv",
        "PRSA_Data_Nongzhanguan_20130301-20170228.csv",
        "PRSA_Data_Shunyi_20130301-20170228.csv",
        "PRSA_Data_Tiantan_20130301-20170228.csv",
        "PRSA_Data_Wanliu_20130301-20170228.csv",
        "PRSA_Data_Wanshouxigong_20130301-20170228.csv",
    ]

    urls = [f"{base_url}/{f}" for f in files]

    df = pd.concat(
        [pd.read_csv(url) for url in urls],
        ignore_index=True
    )

    print(f"Loaded {len(files)} station files combined shape {df.shape}")
    return df

def save_combined(df, out_path="data.csv"):
 
    df.to_csv(out_path, index=False)
    print(f"Saved combined dataset: {out_path}")

if __name__ == "__main__":
    combined = load_all_stations()
    save_combined(combined)