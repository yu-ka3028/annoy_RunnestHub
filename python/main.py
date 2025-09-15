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

print(f"\n=== ベクトル化が完了しました ===")

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

# より詳細なテスト用飲み物リスト
test_drinks = ["lemon_sour", "gin_tonic", "cafe_latte"]
k = 5

print(f"\n🧪 テスト用飲み物での検索実行:")
for drink in test_drinks:
    results = find_similar_drinks_annoy(drink, cosine_annoy_index, k)
    print("\n" + "="*50)

print(f"\n--- Step 2-2: 特定の組み合わせでの類似度分析 ---")

def classify_ingredients(ingredients_list):
    """
    材料を分類する関数
    """
    # 分類定義
    alcohol_types = {
        '蒸留酒': ['whisky', 'shochu', 'gin', 'rum', 'vodka', 'tequila'],
        '醸造酒': ['beer_malt', 'wine', 'sake'],
        'リキュール': ['peach_liqueur', 'orange_liqueur', 'coffee_liqueur', 'almond_liqueur']
    }
    
    carbonated = ['soda', 'tonic_water', 'carbonic_acid']
    fruits = ['lemon', 'orange', 'peach', 'lime', 'grapefruit']
    dairy = ['milk', 'cream', 'yogurt']
    caffeine = ['coffee', 'tea', 'oolong_tea', 'green_tea']
    
    classifications = {
        '酒類': [],
        '炭酸': [],
        '果実': [],
        '乳製品': [],
        'カフェイン': [],
        'その他': []
    }
    
    for ingredient in ingredients_list:
        classified = False
        
        # 酒類の分類
        for category, types in alcohol_types.items():
            if ingredient in types:
                classifications['酒類'].append(f"{ingredient}({category})")
                classified = True
                break
        
        if not classified:
            if ingredient in carbonated:
                classifications['炭酸'].append(ingredient)
                classified = True
            elif ingredient in fruits:
                classifications['果実'].append(ingredient)
                classified = True
            elif ingredient in dairy:
                classifications['乳製品'].append(ingredient)
                classified = True
            elif ingredient in caffeine:
                classifications['カフェイン'].append(ingredient)
                classified = True
            else:
                classifications['その他'].append(ingredient)
    
    return classifications

def compare_specific_drinks(drink1, drink2):
    """
    2つの飲み物の類似度を直接比較する関数
    """
    print(f"\n🔍 {drink1} vs {drink2} の類似度分析:")
    
    # 両方の飲み物の情報を取得
    drink1_idx = drinks_df[drinks_df['name'] == drink1].index[0]
    drink2_idx = drinks_df[drinks_df['name'] == drink2].index[0]
    
    print(f"🍹 {drink1}: {drinks_df.iloc[drink1_idx]['ingredients']}")
    print(f"🍹 {drink2}: {drinks_df.iloc[drink2_idx]['ingredients']}")
    
    # 材料の共通性を分析
    ingredients1 = set(drinks_df.iloc[drink1_idx]['ingredients'].split('|'))
    ingredients2 = set(drinks_df.iloc[drink2_idx]['ingredients'].split('|'))
    common_ingredients = ingredients1 & ingredients2
    unique1 = ingredients1 - ingredients2
    unique2 = ingredients2 - ingredients1
    
    print(f"📊 材料分析:")
    print(f"   共通材料: {list(common_ingredients) if common_ingredients else 'なし'} ({len(common_ingredients)}個)")
    print(f"   {drink1}のみ: {list(unique1)}")
    print(f"   {drink2}のみ: {list(unique2)}")
    
    # 材料の分類表示
    print(f"\n🏷️ 材料分類:")
    for drink_name, ingredients_set in [(drink1, ingredients1), (drink2, ingredients2)]:
        classifications = classify_ingredients(list(ingredients_set))
        print(f"   {drink_name}:")
        for category, items in classifications.items():
            if items:
                print(f"     {category}: {', '.join(items)}")
    
    # 類似度を計算
    vector1 = ingredients_matrix[drink1_idx].toarray()[0]
    vector2 = ingredients_matrix[drink2_idx].toarray()[0]
    
    # コサイン類似度を直接計算
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    cosine_similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
    
    print(f"\n📈 類似度:")
    print(f"   コサイン類似度: {cosine_similarity:.4f}")
    print(f"   期待値: {'高い' if len(common_ingredients) > 0 else '低い'}")
    
    return {
        'drink1': drink1,
        'drink2': drink2,
        'common_ingredients': len(common_ingredients),
        'cosine_similarity': cosine_similarity
    }

# 特定の組み合わせをテスト
test_combinations = [
    ("stout_beer", "draft_beer"),      # 同じ材料
    ("cafe_latte", "energy_drink"),    # カフェイン系
    ("lemon_sour", "highball_lemon"),  # レモン系
    ("gin_tonic", "orange_mojito"),    # 炭酸系
    ("peach_oolong", "lemon_sour"),    # 全く異なる
    ("red_wine", "stout_beer"),        # 酒類だが異なる
]

comparison_results = []
for drink1, drink2 in test_combinations:
    result = compare_specific_drinks(drink1, drink2)
    comparison_results.append(result)
    print("\n" + "-" * 60)

# 結果のまとめ
print(f"\n📋 類似度分析のまとめ:")
print(f"{'飲み物1':<15} {'飲み物2':<15} {'共通材料':<8} {'類似度':<8} {'期待値との一致'}")
print("-" * 70)
for result in comparison_results:
    expected = "高い" if result['common_ingredients'] > 0 else "低い"
    actual = "高い" if result['cosine_similarity'] > 0.3 else "低い"
    match = "✅" if (expected == actual) else "❌"
    
    print(f"{result['drink1']:<15} {result['drink2']:<15} {result['common_ingredients']:<8} {result['cosine_similarity']:<8.4f} {match}")

print(f"\n--- Step 2-3: 全飲み物の類似度マトリックス ---")

def create_similarity_matrix():
    """
    全飲み物の類似度マトリックスを作成する関数
    """
    print(f"\n🔍 全飲み物の類似度マトリックス:")
    
    n_drinks = len(drinks_df)
    similarity_matrix = np.zeros((n_drinks, n_drinks))
    
    for i in range(n_drinks):
        for j in range(n_drinks):
            if i != j:
                vector1 = ingredients_matrix[i].toarray()[0]
                vector2 = ingredients_matrix[j].toarray()[0]
                
                dot_product = np.dot(vector1, vector2)
                norm1 = np.linalg.norm(vector1)
                norm2 = np.linalg.norm(vector2)
                cosine_similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
                
                similarity_matrix[i, j] = cosine_similarity
    
    # 最も類似度が高い組み合わせを表示
    print(f"\n📊 最も類似度が高い組み合わせ (上位5位):")
    high_similarity_pairs = []
    
    for i in range(n_drinks):
        for j in range(i+1, n_drinks):
            similarity = similarity_matrix[i, j]
            drink1 = drinks_df.iloc[i]['name']
            drink2 = drinks_df.iloc[j]['name']
            high_similarity_pairs.append((similarity, drink1, drink2))
    
    high_similarity_pairs.sort(reverse=True)
    
    for rank, (similarity, drink1, drink2) in enumerate(high_similarity_pairs[:5], 1):
        print(f"   {rank}. {drink1} ↔ {drink2}: {similarity:.4f}")
    
    return similarity_matrix

similarity_matrix = create_similarity_matrix()

print(f"\n=== annoyインデックスの作成が完了しました ===")

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

print("\n" + "="*50)
print("6-1. Vim使いがよく飲むお酒ランキング")
print("="*50)

def get_vim_users_drink_ranking():
    """
    Vim使い（uses_vim=1）のユーザーがよく飲む飲み物のランキングを作成する関数
    """
    print(f"\n--- Step 1: Vim使いのユーザーを特定 ---")
    
    vim_users = users_df[users_df['uses_vim'] == 1]
    print(f"Vim使いのユーザー数: {len(vim_users)}人")
    print(f"Vim使いのユーザーID: {vim_users['user_id'].tolist()}")
    
    print(f"\nVim使いのユーザー詳細:")
    for _, user in vim_users.iterrows():
        print(f"  ユーザーID {user['user_id']}: {user['gender']}, {user['age']}歳, {user['favorite_lang']}, {user['editor']}")
    
    print(f"\n--- Step 2: Vim使いの飲み物選択を抽出 ---")

    vim_user_ids = vim_users['user_id'].tolist()
    vim_interactions = interactions_df[interactions_df['user_id'].isin(vim_user_ids)]
    print(f"Vim使いの飲み物選択総数: {len(vim_interactions)}回")

    print(f"\nVim使いの飲み物選択詳細:")
    for _, interaction in vim_interactions.iterrows():
        user_info = vim_users[vim_users['user_id'] == interaction['user_id']].iloc[0]
        drink_info = drinks_df[drinks_df['drink_id'] == interaction['item_id']].iloc[0]
        print(f"  ユーザーID {interaction['user_id']} ({user_info['favorite_lang']}) → {drink_info['name']} ({drink_info['category']})")
    
    print(f"\n--- Step 3: 飲み物別の選択回数をカウント ---")

    drink_counts = vim_interactions['item_id'].value_counts().reset_index()
    drink_counts.columns = ['drink_id', 'count']

    drink_ranking = pd.merge(drink_counts, drinks_df, on='drink_id', how='inner')

    drink_ranking = drink_ranking.sort_values('count', ascending=False).reset_index(drop=True)
    
    print(f"\n--- Step 4: ランキング結果を表示 ---")
    
    print(f"\n🏆 Vim使いがよく飲むお酒ランキング:")
    print("=" * 60)
    print(f"{'順位':<4} {'飲み物名':<15} {'カテゴリ':<12} {'選択回数':<8} {'アルコール度数':<10} {'材料'}")
    print("-" * 60)
    
    for rank, (_, row) in enumerate(drink_ranking.iterrows(), 1):
        print(f"{rank:<4} {row['name']:<15} {row['category']:<12} {row['count']:<8} {row['abv']:<10} {row['ingredients']}")

    print(f"\n📊 ランキング統計:")
    print(f"   総選択回数: {drink_ranking['count'].sum()}回")
    print(f"   飲み物種類数: {len(drink_ranking)}種類")
    print(f"   平均選択回数: {drink_ranking['count'].mean():.2f}回")
    print(f"   最多選択回数: {drink_ranking['count'].max()}回")
    print(f"   最少選択回数: {drink_ranking['count'].min()}回")

    print(f"\n📈 カテゴリ別集計:")
    category_stats = drink_ranking.groupby('category').agg({
        'count': ['sum', 'mean', 'count']
    }).round(2)
    category_stats.columns = ['総選択回数', '平均選択回数', '飲み物種類数']
    print(category_stats)

    print(f"\n🍺 アルコール度数別集計:")
    alcohol_stats = drink_ranking.groupby('abv').agg({
        'count': ['sum', 'mean', 'count']
    }).round(2)
    alcohol_stats.columns = ['総選択回数', '平均選択回数', '飲み物種類数']
    print(alcohol_stats)
    
    return drink_ranking

vim_drink_ranking = get_vim_users_drink_ranking()

print(f"\n--- Step 5: ランキング結果の詳細分析 ---")

def analyze_vim_drink_preferences(ranking_df):
    """
    Vim使いの飲み物嗜好を詳細分析する関数
    """
    print(f"\n🔍 Vim使いの飲み物嗜好分析:")

    print(f"\n🥇 上位3位の詳細分析:")
    for rank, (_, row) in enumerate(ranking_df.head(3).iterrows(), 1):
        print(f"\n{rank}位: {row['name']}")
        print(f"   カテゴリ: {row['category']}")
        print(f"   アルコール度数: {row['abv']}%")
        print(f"   材料: {row['ingredients']}")
        print(f"   選択回数: {row['count']}回")

        ingredients = row['ingredients'].split('|')
        print(f"   材料数: {len(ingredients)}個")
        print(f"   材料詳細: {', '.join(ingredients)}")

    alcohol_drinks = ranking_df[ranking_df['abv'] > 0]
    non_alcohol_drinks = ranking_df[ranking_df['abv'] == 0]
    
    print(f"\n🍻 アルコール系 vs ノンアルコール系:")
    print(f"   アルコール系: {len(alcohol_drinks)}種類, 総選択回数 {alcohol_drinks['count'].sum()}回")
    print(f"   ノンアルコール系: {len(non_alcohol_drinks)}種類, 総選択回数 {non_alcohol_drinks['count'].sum()}回")
    
    if len(alcohol_drinks) > 0:
        print(f"   アルコール系平均選択回数: {alcohol_drinks['count'].mean():.2f}回")
    if len(non_alcohol_drinks) > 0:
        print(f"   ノンアルコール系平均選択回数: {non_alcohol_drinks['count'].mean():.2f}回")

    most_popular_category = ranking_df.groupby('category')['count'].sum().idxmax()
    most_popular_count = ranking_df.groupby('category')['count'].sum().max()
    print(f"\n🏆 最も人気のカテゴリ: {most_popular_category} ({most_popular_count}回選択)")

    all_ingredients = []
    for ingredients_str in ranking_df['ingredients']:
        ingredients = ingredients_str.split('|')
        all_ingredients.extend(ingredients)
    
    from collections import Counter
    ingredient_counts = Counter(all_ingredients)
    print(f"\n🥤 頻出材料トップ5:")
    for ingredient, count in ingredient_counts.most_common(5):
        print(f"   {ingredient}: {count}回使用")

analyze_vim_drink_preferences(vim_drink_ranking)

print(f"\n=== Vim使いがよく飲むお酒ランキングの集計が完了しました ===")

print("\n" + "="*50)
print("6-2. 属性別ランキングの汎用化")
print("="*50)

def get_available_attributes():
    """
    利用可能な属性とその値を取得する関数
    """
    print(f"\n📋 利用可能な属性一覧:")
    print("-" * 50)
    
    attributes = {
        'gender': {'name': '性別', 'values': users_df['gender'].unique()},
        'favorite_lang': {'name': '好きな言語', 'values': users_df['favorite_lang'].unique()},
        'os': {'name': 'OS', 'values': users_df['os'].unique()},
        'editor': {'name': 'エディタ', 'values': users_df['editor'].unique()},
        'night_owl': {'name': '夜型', 'values': users_df['night_owl'].unique()},
        'extroversion_tag': {'name': '性格タイプ', 'values': users_df['extroversion_tag'].unique()},
        'ai_assistant': {'name': 'AIアシスタント', 'values': users_df['ai_assistant'].unique()},
        'uses_vim': {'name': 'Vim使用', 'values': users_df['uses_vim'].unique()}
    }
    
    for i, (attr_key, attr_info) in enumerate(attributes.items(), 1):
        print(f"{i}. {attr_info['name']} ({attr_key})")
        print(f"   利用可能な値: {', '.join(map(str, attr_info['values']))}")
        print()
    
    return attributes

def get_user_filters():
    """
    ユーザーからフィルタ条件を対話式で取得する関数
    """
    attributes = get_available_attributes()
    filters = {}
    
    print(f"🔍 フィルタ条件を設定してください:")
    print(f"複数の条件を設定できます。終了するには 'done' と入力してください。")
    print("-" * 50)
    
    while True:
        print(f"\n現在のフィルタ条件: {filters if filters else 'なし'}")
        
        try:
            attr_choice = input(f"属性を選択してください (1-{len(attributes)}): ").strip()
            
            if attr_choice.lower() == 'done':
                break
                
            attr_choice = int(attr_choice)
            if attr_choice < 1 or attr_choice > len(attributes):
                print(f"❌ 1から{len(attributes)}の間で入力してください")
                continue
                
            attr_key = list(attributes.keys())[attr_choice - 1]
            attr_info = attributes[attr_key]
            
            print(f"\n選択した属性: {attr_info['name']} ({attr_key})")
            print(f"利用可能な値: {', '.join(map(str, attr_info['values']))}")
            
            attr_value = input(f"値を入力してください: ").strip()
            
            if attr_value not in map(str, attr_info['values']):
                print(f"❌ 無効な値です。利用可能な値: {', '.join(map(str, attr_info['values']))}")
                continue

            if attr_key in ['night_owl', 'uses_vim']:
                attr_value = int(attr_value)
            
            filters[attr_key] = attr_value
            print(f"✅ フィルタ条件を追加: {attr_info['name']} = {attr_value}")
            
        except ValueError:
            print(f"❌ 数値を入力してください")
            continue
        except KeyboardInterrupt:
            print(f"\n\n👋 処理を中断しました")
            return None
    
    return filters if filters else None

def get_drink_ranking_by_filters(filters):
    """
    指定されたフィルタ条件で飲み物ランキングを作成する関数
    """
    print(f"\n--- Step 1: フィルタ条件でユーザーを抽出 ---")

    filtered_users = users_df.copy()
    
    for attr_key, attr_value in filters.items():
        filtered_users = filtered_users[filtered_users[attr_key] == attr_value]
        print(f"   {attr_key} = {attr_value}: {len(filtered_users)}人")
    
    if len(filtered_users) == 0:
        print(f"❌ 条件に合致するユーザーが見つかりませんでした")
        return None
    
    print(f"✅ フィルタ後のユーザー数: {len(filtered_users)}人")
    print(f"フィルタ後のユーザーID: {filtered_users['user_id'].tolist()}")
    
    print(f"\n--- Step 2: フィルタ後のユーザーの飲み物選択を抽出 ---")

    filtered_user_ids = filtered_users['user_id'].tolist()
    filtered_interactions = interactions_df[interactions_df['user_id'].isin(filtered_user_ids)]
    print(f"フィルタ後のユーザーの飲み物選択総数: {len(filtered_interactions)}回")
    
    if len(filtered_interactions) == 0:
        print(f"❌ 条件に合致するユーザーの飲み物選択が見つかりませんでした")
        return None
    
    print(f"\n--- Step 3: 飲み物別の選択回数をカウント ---")

    drink_counts = filtered_interactions['item_id'].value_counts().reset_index()
    drink_counts.columns = ['drink_id', 'count']

    drink_ranking = pd.merge(drink_counts, drinks_df, on='drink_id', how='inner')

    drink_ranking = drink_ranking.sort_values('count', ascending=False).reset_index(drop=True)
    
    print(f"\n--- Step 4: ランキング結果を表示 ---")

    filter_description = ", ".join([f"{attr_key}={value}" for attr_key, value in filters.items()])
    
    print(f"\n🏆 フィルタ条件 '{filter_description}' での飲み物ランキング:")
    print("=" * 80)
    print(f"{'順位':<4} {'飲み物名':<15} {'カテゴリ':<12} {'選択回数':<8} {'アルコール度数':<10} {'材料'}")
    print("-" * 80)
    
    for rank, (_, row) in enumerate(drink_ranking.iterrows(), 1):
        print(f"{rank:<4} {row['name']:<15} {row['category']:<12} {row['count']:<8} {row['abv']:<10} {row['ingredients']}")

    print(f"\n📊 ランキング統計:")
    print(f"   フィルタ条件: {filter_description}")
    print(f"   対象ユーザー数: {len(filtered_users)}人")
    print(f"   総選択回数: {drink_ranking['count'].sum()}回")
    print(f"   飲み物種類数: {len(drink_ranking)}種類")
    print(f"   平均選択回数: {drink_ranking['count'].mean():.2f}回")
    print(f"   最多選択回数: {drink_ranking['count'].max()}回")
    print(f"   最少選択回数: {drink_ranking['count'].min()}回")
    
    return drink_ranking, filters

def interactive_drink_ranking():
    """
    対話式で飲み物ランキングを作成するメイン関数
    """
    print(f"\n🎯 属性別飲み物ランキング作成ツール")
    print(f"このツールでは、ユーザーの属性に基づいて飲み物のランキングを作成できます。")
    
    while True:
        print(f"\n" + "="*60)
        print(f"メニュー:")
        print(f"1. 新しいランキングを作成")
        print(f"2. 利用可能な属性を確認")
        print(f"3. 終了")
        
        try:
            choice = input(f"\n選択してください (1-3): ").strip()
            
            if choice == '1':

                filters = get_user_filters()
                
                if filters is None:
                    print(f"フィルタ条件が設定されませんでした。")
                    continue

                result = get_drink_ranking_by_filters(filters)
                
                if result is None:
                    continue
                
                ranking_df, filters = result
                
            elif choice == '2':
                get_available_attributes()
                
            elif choice == '3':
                print(f"\n👋 ツールを終了します。お疲れ様でした！")
                break
                
            else:
                print(f"❌ 1から3の間で入力してください")
                
        except KeyboardInterrupt:
            print(f"\n\n👋 処理を中断しました。ツールを終了します。")
            break
        except Exception as e:
            print(f"❌ エラーが発生しました: {e}")

interactive_drink_ranking()

print(f"\n=== 属性別ランキングの汎用化が完了しました ===")

print("\n" + "="*50)
print("6-3. 統計情報の表示（SP:1）")
print("="*50)

def display_user_attribute_statistics():
    """
    ユーザー属性の分布統計を表示する関数
    """
    print(f"\n📊 ユーザー属性の分布統計")
    print("=" * 60)
    
    print(f"\n🔢 基本統計:")
    print(f"   総ユーザー数: {len(users_df)}人")
    print(f"   平均年齢: {users_df['age'].mean():.1f}歳")
    print(f"   年齢の範囲: {users_df['age'].min()}歳 ～ {users_df['age'].max()}歳")
    print(f"   平均コーディング時間: {users_df['coding_hours_per_day'].mean():.1f}時間/日")
    
    print(f"\n👥 性別分布:")
    gender_counts = users_df['gender'].value_counts()
    for gender, count in gender_counts.items():
        percentage = (count / len(users_df)) * 100
        print(f"   {gender}: {count}人 ({percentage:.1f}%)")
    
    print(f"\n🎂 年齢分布:")
    age_stats = users_df['age'].describe()
    print(f"   平均: {age_stats['mean']:.1f}歳")
    print(f"   中央値: {age_stats['50%']:.1f}歳")
    print(f"   標準偏差: {age_stats['std']:.1f}歳")
    print(f"   最小値: {age_stats['min']:.0f}歳")
    print(f"   最大値: {age_stats['max']:.0f}歳")
    print(f"   25%分位: {age_stats['25%']:.1f}歳")
    print(f"   75%分位: {age_stats['75%']:.1f}歳")
    
    print(f"\n💻 好きな言語分布:")
    lang_counts = users_df['favorite_lang'].value_counts()
    for lang, count in lang_counts.items():
        percentage = (count / len(users_df)) * 100
        print(f"   {lang}: {count}人 ({percentage:.1f}%)")
    
    print(f"\n🖥️ OS分布:")
    os_counts = users_df['os'].value_counts()
    for os, count in os_counts.items():
        percentage = (count / len(users_df)) * 100
        print(f"   {os}: {count}人 ({percentage:.1f}%)")
    
    print(f"\n✏️ エディタ分布:")
    editor_counts = users_df['editor'].value_counts()
    for editor, count in editor_counts.items():
        percentage = (count / len(users_df)) * 100
        print(f"   {editor}: {count}人 ({percentage:.1f}%)")
    
    print(f"\n🌙 夜型分布:")
    night_owl_counts = users_df['night_owl'].value_counts()
    for night_owl, count in night_owl_counts.items():
        percentage = (count / len(users_df)) * 100
        type_name = "夜型" if night_owl == 1 else "朝型"
        print(f"   {type_name}: {count}人 ({percentage:.1f}%)")
    
    print(f"\n🧠 性格タイプ分布:")
    personality_counts = users_df['extroversion_tag'].value_counts()
    for personality, count in personality_counts.items():
        percentage = (count / len(users_df)) * 100
        print(f"   {personality}: {count}人 ({percentage:.1f}%)")
    
    print(f"\n🤖 AIアシスタント使用分布:")
    ai_counts = users_df['ai_assistant'].value_counts()
    for ai, count in ai_counts.items():
        percentage = (count / len(users_df)) * 100
        print(f"   {ai}: {count}人 ({percentage:.1f}%)")
    
    print(f"\n⌨️ Vim使用分布:")
    vim_counts = users_df['uses_vim'].value_counts()
    for vim, count in vim_counts.items():
        percentage = (count / len(users_df)) * 100
        type_name = "Vim使用" if vim == 1 else "Vim未使用"
        print(f"   {type_name}: {count}人 ({percentage:.1f}%)")
    
    print(f"\n🍺 お気に入りアルコール分布:")
    alcohol_counts = users_df['favorite_alcohol'].value_counts()
    for alcohol, count in alcohol_counts.items():
        percentage = (count / len(users_df)) * 100
        print(f"   {alcohol}: {count}人 ({percentage:.1f}%)")
    
    return {
        'total_users': len(users_df),
        'age_stats': age_stats,
        'gender_distribution': gender_counts,
        'language_distribution': lang_counts,
        'os_distribution': os_counts,
        'editor_distribution': editor_counts,
        'night_owl_distribution': night_owl_counts,
        'personality_distribution': personality_counts,
        'ai_assistant_distribution': ai_counts,
        'vim_usage_distribution': vim_counts,
        'favorite_alcohol_distribution': alcohol_counts
    }

def display_drink_category_statistics():
    """
    飲み物カテゴリの分布統計を表示する関数
    """
    print(f"\n🍹 飲み物カテゴリの分布統計")
    print("=" * 60)
    
    print(f"\n🔢 基本統計:")
    print(f"   総飲み物数: {len(drinks_df)}種類")
    print(f"   平均アルコール度数: {drinks_df['abv'].mean():.1f}%")
    print(f"   アルコール度数の範囲: {drinks_df['abv'].min()}% ～ {drinks_df['abv'].max()}%")

    print(f"\n📂 カテゴリ分布:")
    category_counts = drinks_df['category'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(drinks_df)) * 100
        print(f"   {category}: {count}種類 ({percentage:.1f}%)")

    print(f"\n🍺 アルコール度数分布:")
    abv_stats = drinks_df['abv'].describe()
    print(f"   平均: {abv_stats['mean']:.1f}%")
    print(f"   中央値: {abv_stats['50%']:.1f}%")
    print(f"   標準偏差: {abv_stats['std']:.1f}%")
    print(f"   最小値: {abv_stats['min']:.0f}%")
    print(f"   最大値: {abv_stats['max']:.0f}%")

    print(f"\n🍻 アルコール有無分布:")
    alcohol_drinks = len(drinks_df[drinks_df['abv'] > 0])
    non_alcohol_drinks = len(drinks_df[drinks_df['abv'] == 0])
    total_drinks = len(drinks_df)
    
    print(f"   アルコール系: {alcohol_drinks}種類 ({(alcohol_drinks/total_drinks)*100:.1f}%)")
    print(f"   ノンアルコール系: {non_alcohol_drinks}種類 ({(non_alcohol_drinks/total_drinks)*100:.1f}%)")

    print(f"\n🥤 材料数分布:")
    drinks_df['ingredient_count'] = drinks_df['ingredients'].str.split('|').str.len()
    ingredient_stats = drinks_df['ingredient_count'].describe()
    print(f"   平均材料数: {ingredient_stats['mean']:.1f}個")
    print(f"   中央値: {ingredient_stats['50%']:.1f}個")
    print(f"   標準偏差: {ingredient_stats['std']:.1f}個")
    print(f"   最小値: {ingredient_stats['min']:.0f}個")
    print(f"   最大値: {ingredient_stats['max']:.0f}個")

    print(f"\n🥄 材料の頻出度（上位10位）:")
    all_ingredients = []
    for ingredients_str in drinks_df['ingredients']:
        ingredients = ingredients_str.split('|')
        all_ingredients.extend(ingredients)
    
    from collections import Counter
    ingredient_counts = Counter(all_ingredients)
    for i, (ingredient, count) in enumerate(ingredient_counts.most_common(10), 1):
        percentage = (count / len(drinks_df)) * 100
        print(f"   {i:2d}. {ingredient}: {count}回使用 ({percentage:.1f}%)")
    
    return {
        'total_drinks': len(drinks_df),
        'abv_stats': abv_stats,
        'category_distribution': category_counts,
        'alcohol_vs_non_alcohol': {'alcohol': alcohol_drinks, 'non_alcohol': non_alcohol_drinks},
        'ingredient_count_stats': ingredient_stats,
        'top_ingredients': ingredient_counts.most_common(10)
    }

def display_interaction_statistics():
    """
    インタラクション（飲み物選択）の統計を表示する関数
    """
    print(f"\n🍻 インタラクション統計")
    print("=" * 60)

    print(f"\n🔢 基本統計:")
    print(f"   総インタラクション数: {len(interactions_df)}回")
    print(f"   平均ユーザーあたりの選択回数: {len(interactions_df) / len(users_df):.1f}回")
    print(f"   平均飲み物あたりの選択回数: {len(interactions_df) / len(drinks_df):.1f}回")

    print(f"\n👤 ユーザー別選択回数:")
    user_interaction_counts = interactions_df['user_id'].value_counts()
    user_stats = user_interaction_counts.describe()
    print(f"   平均選択回数: {user_stats['mean']:.1f}回")
    print(f"   中央値: {user_stats['50%']:.1f}回")
    print(f"   標準偏差: {user_stats['std']:.1f}回")
    print(f"   最小値: {user_stats['min']:.0f}回")
    print(f"   最大値: {user_stats['max']:.0f}回")

    print(f"\n🍹 飲み物別選択回数:")
    drink_interaction_counts = interactions_df['item_id'].value_counts()
    drink_stats = drink_interaction_counts.describe()
    print(f"   平均選択回数: {drink_stats['mean']:.1f}回")
    print(f"   中央値: {drink_stats['50%']:.1f}回")
    print(f"   標準偏差: {drink_stats['std']:.1f}回")
    print(f"   最小値: {drink_stats['min']:.0f}回")
    print(f"   最大値: {drink_stats['max']:.0f}回")

    print(f"\n🏆 人気飲み物トップ5:")
    for i, (drink_id, count) in enumerate(drink_interaction_counts.head(5).items(), 1):
        drink_name = drinks_df[drinks_df['drink_id'] == drink_id]['name'].iloc[0]
        percentage = (count / len(interactions_df)) * 100
        print(f"   {i}. {drink_name}: {count}回選択 ({percentage:.1f}%)")
    
    return {
        'total_interactions': len(interactions_df),
        'user_interaction_stats': user_stats,
        'drink_interaction_stats': drink_stats,
        'top_drinks': drink_interaction_counts.head(5)
    }


def display_comprehensive_statistics():
    """
    包括的な統計情報を表示するメイン関数
    """
    print(f"\n🎯 包括的統計情報の表示")
    print("=" * 80)

    user_stats = display_user_attribute_statistics()
    drink_stats = display_drink_category_statistics()
    interaction_stats = display_interaction_statistics()

    print(f"\n🔍 データ品質チェック")
    print("=" * 60)

    print(f"\n❓ 欠損値チェック:")
    for df_name, df in [('users', users_df), ('drinks', drinks_df), ('interactions', interactions_df)]:
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"   {df_name}: {missing_count}個の欠損値あり")
        else:
            print(f"   {df_name}: 欠損値なし ✅")

    print(f"\n🔗 データ整合性チェック:")

    user_ids_in_interactions = set(interactions_df['user_id'].unique())
    user_ids_in_users = set(users_df['user_id'].unique())
    missing_users = user_ids_in_interactions - user_ids_in_users
    if missing_users:
        print(f"   ❌ インタラクションに存在するがユーザーテーブルにないユーザーID: {missing_users}")
    else:
        print(f"   ✅ ユーザーIDの整合性: 問題なし")

    drink_ids_in_interactions = set(interactions_df['item_id'].unique())
    drink_ids_in_drinks = set(drinks_df['drink_id'].unique())
    missing_drinks = drink_ids_in_interactions - drink_ids_in_drinks
    if missing_drinks:
        print(f"   ❌ インタラクションに存在するが飲み物テーブルにない飲み物ID: {missing_drinks}")
    else:
        print(f"   ✅ 飲み物IDの整合性: 問題なし")

    print(f"\n📋 データ分布の要約")
    print("=" * 60)
    print(f"   ユーザー数: {len(users_df)}人")
    print(f"   飲み物数: {len(drinks_df)}種類")
    print(f"   インタラクション数: {len(interactions_df)}回")
    print(f"   データ密度: {len(interactions_df) / (len(users_df) * len(drinks_df)) * 100:.1f}%")
    
    return {
        'user_stats': user_stats,
        'drink_stats': drink_stats,
        'interaction_stats': interaction_stats
    }

comprehensive_stats = display_comprehensive_statistics()

print(f"\n=== 統計情報の表示が完了しました ===")
