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

print("\n" + "="*50)
print("5-1. annoyで似ている飲み物の近似K近傍検索")
print("="*50)

import time
from annoy import AnnoyIndex

print(f"\n--- Step 1: annoyライブラリの確認とインデックス作成 ---")

try:
    print(f"✅ annoyライブラリのインポート成功")
except ImportError:
    print(f"❌ エラー: annoyライブラリがインストールされていません")
    print(f"💡 pip install annoy でインストールしてください")
    exit(1)

# カラム名でshapeの意味を確認
feature_names = vectorizer.get_feature_names_out()
print(f"\n🔍 カラム名でshapeの意味を確認:")
print(f"   shape[0]のカラム名: '{drinks_df.columns[0]}' ← 飲み物名（行）")
print(f"   shape[1]のカラム名: 'feature_names' ← 特徴量名（列）")
print(f"   → 実際の特徴量名: {list(feature_names)}")
print(f"   → shape[0]は飲み物、shape[1]は材料（特徴量）を表していることが確認できます")
print(f"   → 実際のカラム名: shape[0]='{drinks_df.columns[0]}', shape[1]='feature_names'")
print(f"   → 特徴量の例: {feature_names[:5]}... (全{len(feature_names)}個)")

# ベクトルの次元数を取得
print(f"📊 ingredients_matrixのshape: {ingredients_matrix.shape}")
print(f"📊 shape[0] (行数): {ingredients_matrix.shape[0]} ← 飲み物の数")
print(f"📊 shape[1] (列数): {ingredients_matrix.shape[1]} ← 特徴量の数")

vector_dimension = ingredients_matrix.shape[1]
print(f"📊 ベクトルの次元数: {vector_dimension}")
print(f"📊 飲み物数: {ingredients_matrix.shape[0]}")

print(f"\n🔧 annoyインデックスを作成中...")
cosine_annoy_index = AnnoyIndex(vector_dimension, 'angular')

print(f"📝 インデックスへのベクトル追加を開始...")

for i in range(ingredients_matrix.shape[0]):
    vector = ingredients_matrix[i].toarray()[0]
    drink_name = drinks_df.iloc[i]['name']
    print(f"   処理中: i={i} → {drink_name} → ベクトル長={len(vector)}")
    
    # annoyインデックスに追加（i番目の飲み物のベクトルを登録）
    cosine_annoy_index.add_item(i, vector)
    
    if (i + 1) % 5 == 0 or i == ingredients_matrix.shape[0] - 1:
        print(f"   進捗: {i + 1}/{ingredients_matrix.shape[0]} ベクトル追加完了")

print(f"✅ インデックスへのベクトル追加が完了しました")

# インデックスのビルド
print(f"\n🏗️ インデックスのビルドを開始...")
n_trees = 10  # ツリー数（精度と速度のバランス）
cosine_annoy_index.build(n_trees)
print(f"✅ インデックスのビルドが完了しました (n_trees={n_trees})")

print(f"\n--- Step 2: annoyでの近傍検索 ---")

def find_similar_drinks_annoy(drink_name, annoy_index, k=5):
    """
    annoyを使って指定された飲み物に似ている飲み物を検索する関数
    """
    # 飲み物名からインデックスを取得
    drink_index = drinks_df[drinks_df['name'] == drink_name].index
    if len(drink_index) == 0:
        print(f"❌ エラー: '{drink_name}' という飲み物が見つかりません")
        return None
    
    drink_idx = drink_index[0]
    print(f"\n🔍 検索対象: {drink_name} (インデックス: {drink_idx})")
    print(f"🍹 材料: {drinks_df.iloc[drink_idx]['ingredients']}")
    
    query_vector = ingredients_matrix[drink_idx].toarray()[0]
    indices, distances = annoy_index.get_nns_by_vector(query_vector, k+1, include_distances=True)
    
    print(f"\n📋 類似飲み物 (K={k}) [annoy コサイン類似度]:")
    print("-" * 70)
    results = []
    for i in range(1, len(indices)):  # 最初の結果（自分自身）をスキップ
        idx = indices[i]
        distance = distances[i]
        similarity = 1 - distance  # コサイン距離を類似度に変換
        similar_drink = drinks_df.iloc[idx]
        
        result = {
            'rank': i,
            'drink_name': similar_drink['name'],
            'similarity': similarity,
            'distance': distance,
            'ingredients': similar_drink['ingredients'],
            'category': similar_drink['category'],
            'abv': similar_drink['abv']
        }
        results.append(result)
        
        print(f"{i:2d}. {similar_drink['name']:15s} | 類似度: {similarity:.4f} | カテゴリ: {similar_drink['category']:10s} | 材料: {similar_drink['ingredients']}")
    
    return results

test_drinks = ["lemon_sour", "gin_tonic", "cafe_latte"]
k = 5

print(f"\n🧪 テスト用飲み物での検索実行:")
for drink in test_drinks:
    results = find_similar_drinks_annoy(drink, cosine_annoy_index, k)
    print("\n" + "="*50)

print(f"\n--- Step 3: インデックスの保存と読み込み ---")

annoy_save_dir = "saved_annoy_indexes"
os.makedirs(annoy_save_dir, exist_ok=True)

cosine_annoy_path = os.path.join(annoy_save_dir, "cosine_annoy_index.ann")

cosine_annoy_index.save(cosine_annoy_path)
print(f"💾 コサイン類似度インデックスを保存: {cosine_annoy_path}")

if os.path.exists(cosine_annoy_path):
    file_size = os.path.getsize(cosine_annoy_path)
    print(f"📁 ファイルサイズ: {file_size:,} bytes ({file_size/1024:.2f} KB)")

print(f"\n🔄 保存したインデックスの読み込みテスト...")

loaded_cosine_annoy = AnnoyIndex(vector_dimension, 'angular')
loaded_cosine_annoy.load(cosine_annoy_path)
print(f"✅ インデックスの読み込みが完了しました")

test_drink = "lemon_sour"
test_idx = drinks_df[drinks_df['name'] == test_drink].index[0]
test_vector = ingredients_matrix[test_idx].toarray()[0]

print(f"\n🧪 読み込みテスト: {test_drink}の検索")
original_results = cosine_annoy_index.get_nns_by_vector(test_vector, 3, include_distances=True)
loaded_results = loaded_cosine_annoy.get_nns_by_vector(test_vector, 3, include_distances=True)

print(f"🔍 元のインデックス結果: {original_results[0]}")
print(f"🔍 読み込みインデックス結果: {loaded_results[0]}")
print(f"✅ 結果が一致: {original_results[0] == loaded_results[0]}")

print(f"📊 距離値の比較:")
for i in range(len(original_results[1])):
    orig_dist = original_results[1][i]
    loaded_dist = loaded_results[1][i]
    diff = abs(orig_dist - loaded_dist)
    print(f"   結果{i+1}: 元={orig_dist:.6f}, 読み込み={loaded_dist:.6f}, 差分={diff:.6f}")

print(f"\n--- Step 4: 検索精度と速度の比較 ---")

def measure_search_performance(annoy_index, test_drinks, k=5, num_tests=100):
    """
    annoyの検索性能を測定する関数
    """
    print(f"\n⏱️ 検索性能測定開始 (テスト回数: {num_tests})")
    
    search_times = []
    all_results = []
    
    for test_num in range(num_tests):
        for drink in test_drinks:

            drink_index = drinks_df[drinks_df['name'] == drink].index
            if len(drink_index) == 0:
                continue
            
            drink_idx = drink_index[0]
            query_vector = ingredients_matrix[drink_idx].toarray()[0]
            

            start_time = time.time()
            indices, distances = annoy_index.get_nns_by_vector(query_vector, k+1, include_distances=True)
            end_time = time.time()
            
            search_time = (end_time - start_time) * 1000
            search_times.append(search_time)
            
            if test_num == 0:
                results = []
                for i in range(1, len(indices)):  # 自分自身を除く
                    idx = indices[i]
                    distance = distances[i]
                    similarity = 1 - distance
                    similar_drink = drinks_df.iloc[idx]
                    
                    results.append({
                        'query_drink': drink,
                        'rank': i,
                        'drink_name': similar_drink['name'],
                        'similarity': similarity,
                        'distance': distance,
                        'ingredients': similar_drink['ingredients'],
                        'category': similar_drink['category']
                    })
                all_results.extend(results)
    
    # 統計情報の計算
    avg_time = np.mean(search_times)
    min_time = np.min(search_times)
    max_time = np.max(search_times)
    std_time = np.std(search_times)
    
    print(f"📊 検索時間統計:")
    print(f"   平均時間: {avg_time:.4f} ms")
    print(f"   最小時間: {min_time:.4f} ms")
    print(f"   最大時間: {max_time:.4f} ms")
    print(f"   標準偏差: {std_time:.4f} ms")
    print(f"   総検索回数: {len(search_times)}回")
    
    return {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'std_time': std_time,
        'total_searches': len(search_times),
        'results': all_results
    }

def analyze_search_accuracy(results):
    """
    検索精度を分析する関数
    """
    print(f"\n🎯 検索精度分析:")
    
    query_analysis = {}
    
    for result in results:
        query_drink = result['query_drink']
        if query_drink not in query_analysis:
            query_analysis[query_drink] = []
        query_analysis[query_drink].append(result)
    
    print(f"📋 クエリドリンク別の検索結果:")
    for query_drink, query_results in query_analysis.items():
        print(f"\n🍹 {query_drink}:")
        print(f"   材料: {drinks_df[drinks_df['name'] == query_drink]['ingredients'].iloc[0]}")
        
        for result in query_results[:3]:
            print(f"   {result['rank']}. {result['drink_name']:15s} | 類似度: {result['similarity']:.4f} | カテゴリ: {result['category']:10s}")
        
        query_category = drinks_df[drinks_df['name'] == query_drink]['category'].iloc[0]
        same_category_count = sum(1 for r in query_results if r['category'] == query_category)
        print(f"   📊 同じカテゴリ({query_category})の飲み物: {same_category_count}/{len(query_results)}件")

# 性能測定実行
performance_stats = measure_search_performance(cosine_annoy_index, test_drinks, k=5, num_tests=50)

# 精度分析実行
analyze_search_accuracy(performance_stats['results'])

print(f"\n--- Step 5: 異なるK値での性能比較 ---")

def compare_different_k_values(annoy_index, test_drink, k_values=[3, 5, 7, 10]):
    """
    異なるK値での検索性能と結果を比較する関数
    """
    print(f"\n🔍 {test_drink} の異なるK値での検索比較:")
    
    drink_idx = drinks_df[drinks_df['name'] == test_drink].index[0]
    query_vector = ingredients_matrix[drink_idx].toarray()[0]
    print(f"🍹 材料: {drinks_df.iloc[drink_idx]['ingredients']}")
    print(f"🏷️ カテゴリ: {drinks_df.iloc[drink_idx]['category']}")
    
    results_comparison = []
    
    for k in k_values:
        times = []
        for _ in range(20):
            start_time = time.time()
            indices, distances = annoy_index.get_nns_by_vector(query_vector, k+1, include_distances=True)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
        
        avg_time = np.mean(times)
        
        k_results = []
        for i in range(1, len(indices)):  # 自分自身を除く
            idx = indices[i]
            distance = distances[i]
            similarity = 1 - distance
            similar_drink = drinks_df.iloc[idx]
            
            k_results.append({
                'k': k,
                'rank': i,
                'drink_name': similar_drink['name'],
                'similarity': similarity,
                'category': similar_drink['category']
            })
        
        results_comparison.append({
            'k': k,
            'avg_time': avg_time,
            'results': k_results
        })
        
        print(f"\n📊 K={k}: 平均検索時間 {avg_time:.4f} ms")
        for i, result in enumerate(k_results[:3]):
            print(f"   {result['rank']}. {result['drink_name']:15s} | 類似度: {result['similarity']:.4f} | カテゴリ: {result['category']:10s}")
    
    return results_comparison

k_comparison = compare_different_k_values(cosine_annoy_index, "lemon_sour", [3, 5, 7, 10])

print(f"\n=== annoyでの近似K近傍検索が完了しました ===")
