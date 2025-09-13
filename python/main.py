# main.py
import pandas as pd
import numpy as np
import os

print("Hello from main.py!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

# CSVファイルのパスを設定
data_dir = "data"
drinks_path = os.path.join(data_dir, "drinks.csv")
users_path = os.path.join(data_dir, "users.csv")
interactions_path = os.path.join(data_dir, "interactions.csv")

print("\n" + "="*50)
print("1. drinks.csvを読み込み")
print("="*50)

# drinks.csvを読み込み
drinks_df = pd.read_csv(drinks_path, comment='#')
print("drinks.csvの内容:")
print(drinks_df)

print(f"\ndrinks.csvのshape: {drinks_df.shape}")
print(f"drinks.csvのcolumns: {list(drinks_df.columns)}")
print("\ndrinks.csvのhead():")
print(drinks_df.head())

print("\n" + "="*50)
print("2. users.csvを読み込み")
print("="*50)

# users.csvを読み込み
users_df = pd.read_csv(users_path, comment='#')
print("users.csvの内容:")
print(users_df)

print(f"\nusers.csvのshape: {users_df.shape}")
print(f"users.csvのcolumns: {list(users_df.columns)}")
print("\nusers.csvのhead():")
print(users_df.head())

print("\n" + "="*50)
print("3. interactions.csvを読み込み")
print("="*50)

# interactions.csvを読み込み
interactions_df = pd.read_csv(interactions_path, comment='#')
print("interactions.csvの内容:")
print(interactions_df)

print(f"\ninteractions.csvのshape: {interactions_df.shape}")
print(f"interactions.csvのcolumns: {list(interactions_df.columns)}")
print("\ninteractions.csvのhead():")
print(interactions_df.head())
