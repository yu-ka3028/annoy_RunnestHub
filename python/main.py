# main.py
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

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

print("\n" + "="*50)
print("3-2. ingredients列を分割してリスト化")
print("="*50)

drinks_df['ingredients_list'] = drinks_df['ingredients'].str.split('|')
print("ingredients列を分割した結果:")
print(drinks_df[['name', 'ingredients', 'ingredients_list']])

print(f"\n分割後のingredients_listの例:")
for i, row in drinks_df.head(3).iterrows():
    print(f"{row['name']}: {row['ingredients_list']}")

print(f"\ningredients_list列のデータ型: {type(drinks_df['ingredients_list'].iloc[0])}")
print(f"各飲み物の材料数:")
for i, row in drinks_df.iterrows():
    print(f"{row['name']}: {len(row['ingredients_list'])}個の材料")

print("\n" + "="*50)
print("3-3. データの結合と前処理")
print("="*50)

print("\n--- Step 1: users.csvとinteractions.csvをuser_idで結合 ---")

user_interactions_df = pd.merge(users_df, interactions_df, on='user_id', how='inner')
print("結合後のデータ:")
print(user_interactions_df)
print(f"\n結合後のshape: {user_interactions_df.shape}")
print(f"結合後のcolumns: {list(user_interactions_df.columns)}")

print("\n--- Step 2: 結合したデータとdrinks.csvをitem_idで結合 ---")

merged_df = pd.merge(user_interactions_df, drinks_df, left_on='item_id', right_on='drink_id', how='inner')
print("最終結合後のデータ:")
print(merged_df)
print(f"\n最終結合後のshape: {merged_df.shape}")
print(f"最終結合後のcolumns: {list(merged_df.columns)}")

print("\n--- Step 3: 欠損値の確認 ---")
print("各列の欠損値数:")
missing_values = merged_df.isnull().sum()
print(missing_values)

print("\n欠損値の詳細:")
for col in merged_df.columns:
    if merged_df[col].isnull().sum() > 0:
        print(f"{col}: {merged_df[col].isnull().sum()}個の欠損値")
        print(f"  欠損値のインデックス: {merged_df[merged_df[col].isnull()].index.tolist()}")

print("\n--- Step 4: データ型の確認 ---")
print("各列のデータ型:")
print(merged_df.dtypes)

print("\n--- Step 5: データ型の変換（必要に応じて） ---")

numeric_columns = ['user_id', 'age', 'uses_vim', 'coding_hours_per_day', 'night_owl', 'item_id', 'drink_id', 'abv']
for col in numeric_columns:
    if col in merged_df.columns:
        print(f"{col}: {merged_df[col].dtype} -> 数値型に変換")
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

categorical_columns = ['gender', 'favorite_lang', 'ai_assistant', 'os', 'editor', 'extroversion_tag', 'favorite_alcohol', 'item_type', 'name', 'category']
for col in categorical_columns:
    if col in merged_df.columns:
        print(f"{col}: {merged_df[col].dtype} -> カテゴリ型に変換")
        merged_df[col] = merged_df[col].astype('category')

print("\n変換後のデータ型:")
print(merged_df.dtypes)

print("\n--- 最終データの確認 ---")
print("結合後のデータの最初の5行:")
print(merged_df.head())
print(f"\n最終データのshape: {merged_df.shape}")
print(f"最終データのメモリ使用量: {merged_df.memory_usage(deep=True).sum() / 1024:.2f} KB")

print("\n" + "="*50)
print("4-1. TfidfVectorizerでingredientsをベクトル化")
print("="*50)

# ingredients_listを文字列に変換（TfidfVectorizer用）
ingredients_text = drinks_df['ingredients_list'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
print("ingredients_textの例:")
for i, text in enumerate(ingredients_text.head(3)):
    print(f"{drinks_df.iloc[i]['name']}: {text}")

# TfidfVectorizerのパラメータ設定
max_features = 20  # 最大特徴量数（重要度の高い20個の材料を特徴量として採用）
min_df = 1         # 最小文書頻度（すべての材料を保持）
max_df = 0.7       # 最大文書頻度（70%以上に出現する単語を除外）

print(f"\nパラメータ設定の理由:")
print(f"- max_features=20: 重要度の高い20個の材料を特徴量として採用")
print(f"- min_df=1: すべての材料を保持（出現頻度で除外しない）")
print(f"- max_df=0.7: 7個以上（70%）の飲み物に出現する単語を除外")

print(f"\nTfidfVectorizerのパラメータ:")
print(f"max_features: {max_features}")
print(f"min_df: {min_df}")
print(f"max_df: {max_df}")

vectorizer = TfidfVectorizer(
    max_features=max_features,
    min_df=min_df,
    max_df=max_df,
    stop_words=None
)

ingredients_matrix = vectorizer.fit_transform(ingredients_text)
print(f"\nベクトル化結果:")
print(f"ベクトル行列のshape: {ingredients_matrix.shape}")
print(f"ベクトル行列の型: {type(ingredients_matrix)}")
print(f"ベクトル行列の密度: {ingredients_matrix.nnz / (ingredients_matrix.shape[0] * ingredients_matrix.shape[1]):.4f}")

print(f"\nベクトル化された内容（最初の3つの飲み物）:")
for i in range(min(3, ingredients_matrix.shape[0])):
    print(f"{drinks_df.iloc[i]['name']}:")
    print(f"  非ゼロ要素数: {ingredients_matrix[i].nnz}")
    print(f"  ベクトル値: {ingredients_matrix[i].toarray()}")

feature_names = vectorizer.get_feature_names_out()
print(f"\n特徴量名（ingredients名）:")
print(f"特徴量数: {len(feature_names)}")
print(f"特徴量名一覧: {list(feature_names)}")
print(f"feature_namesの型: {type(feature_names)}")
print(f"feature_namesのdtype: {feature_names.dtype}")
print(f"feature_namesのshape: {feature_names.shape}")
print(f"\n各特徴量の重要度（TF-IDF値）の統計:")
tfidf_scores = ingredients_matrix.toarray()
for i, feature in enumerate(feature_names):
    feature_scores = tfidf_scores[:, i]
    non_zero_scores = feature_scores[feature_scores > 0]
    if len(non_zero_scores) > 0:
        print(f"{feature}: 平均={non_zero_scores.mean():.4f}, 最大={non_zero_scores.max():.4f}, 非ゼロ数={len(non_zero_scores)}")

print("\n" + "="*50)
print("4-2. ベクトル化結果の保存と読み込み")
print("="*50)

save_dir = "saved_vectors"
os.makedirs(save_dir, exist_ok=True)

print(f"\n--- Step 1: pickleでベクトル化結果を保存 ---")

vectorizer_pickle_path = os.path.join(save_dir, "vectorizer.pkl")
ingredients_matrix_pickle_path = os.path.join(save_dir, "ingredients_matrix.pkl")
feature_names_pickle_path = os.path.join(save_dir, "feature_names.pkl")

with open(vectorizer_pickle_path, 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"vectorizerを保存: {vectorizer_pickle_path}")

with open(ingredients_matrix_pickle_path, 'wb') as f:
    pickle.dump(ingredients_matrix, f)
print(f"ingredients_matrixを保存: {ingredients_matrix_pickle_path}")

with open(feature_names_pickle_path, 'wb') as f:
    pickle.dump(feature_names, f)
print(f"feature_namesを保存: {feature_names_pickle_path}")

print(f"\n--- Step 2: numpyでベクトル化結果を保存 ---")

ingredients_matrix_npy_path = os.path.join(save_dir, "ingredients_matrix.npy")
feature_names_npy_path = os.path.join(save_dir, "feature_names.npy")

np.save(ingredients_matrix_npy_path, ingredients_matrix.toarray())
print(f"ingredients_matrixをnumpy形式で保存: {ingredients_matrix_npy_path}")

np.save(feature_names_npy_path, feature_names)
print(f"feature_namesをnumpy形式で保存: {feature_names_npy_path}")

print(f"\n--- Step 3: 保存したファイルのサイズ確認 ---")
import os.path

files_to_check = [
    vectorizer_pickle_path,
    ingredients_matrix_pickle_path,
    feature_names_pickle_path,
    ingredients_matrix_npy_path,
    feature_names_npy_path
]

for file_path in files_to_check:
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        print(f"{os.path.basename(file_path)}: {file_size:,} bytes ({file_size/1024:.2f} KB)")

print(f"\n--- Step 4: pickleで保存したファイルから読み込み ---")

with open(vectorizer_pickle_path, 'rb') as f:
    loaded_vectorizer = pickle.load(f)
print(f"vectorizerを読み込み完了")

with open(ingredients_matrix_pickle_path, 'rb') as f:
    loaded_ingredients_matrix = pickle.load(f)
print(f"ingredients_matrixを読み込み完了")

with open(feature_names_pickle_path, 'rb') as f:
    loaded_feature_names = pickle.load(f)
print(f"feature_namesを読み込み完了")

print(f"\n読み込み結果の検証:")
print(f"元のvectorizer型: {type(vectorizer)}")
print(f"読み込みvectorizer型: {type(loaded_vectorizer)}")
print(f"元のingredients_matrix shape: {ingredients_matrix.shape}")
print(f"読み込みingredients_matrix shape: {loaded_ingredients_matrix.shape}")
print(f"元のfeature_names shape: {feature_names.shape}")
print(f"読み込みfeature_names shape: {loaded_feature_names.shape}")

print(f"\nデータの一致確認:")
print(f"ingredients_matrixが一致: {np.array_equal(ingredients_matrix.toarray(), loaded_ingredients_matrix.toarray())}")
print(f"feature_namesが一致: {np.array_equal(feature_names, loaded_feature_names)}")

print(f"\n--- Step 5: numpyで保存したファイルから読み込み ---")

loaded_ingredients_matrix_npy = np.load(ingredients_matrix_npy_path)
loaded_feature_names_npy = np.load(feature_names_npy_path, allow_pickle=True)

print(f"numpy形式でingredients_matrixを読み込み完了")
print(f"numpy形式でfeature_namesを読み込み完了")

print(f"\nnumpy読み込み結果の検証:")
print(f"読み込みingredients_matrix shape: {loaded_ingredients_matrix_npy.shape}")
print(f"読み込みfeature_names shape: {loaded_feature_names_npy.shape}")

print(f"\nnumpyデータの一致確認:")
print(f"ingredients_matrixが一致: {np.array_equal(ingredients_matrix.toarray(), loaded_ingredients_matrix_npy)}")
print(f"feature_namesが一致: {np.array_equal(feature_names, loaded_feature_names_npy)}")

print(f"\n--- Step 6: 保存・読み込み機能のテスト ---")

test_text = "コーヒー ミルク 砂糖"
print(f"\nテスト用テキスト: '{test_text}'")

test_vector = loaded_vectorizer.transform([test_text])
print(f"テストベクトルのshape: {test_vector.shape}")
print(f"テストベクトルの値: {test_vector.toarray()}")

test_array = test_vector.toarray()[0]
print(f"\nテストベクトルの詳細:")
for i, (feature, value) in enumerate(zip(loaded_feature_names, test_array)):
    if value > 0:
        print(f"  {feature}: {value:.4f}")

print(f"\n=== ベクトル化結果の保存と読み込みが完了しました ===")
