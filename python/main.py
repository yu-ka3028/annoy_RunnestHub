# main.py - 同人誌用最小出力版
import pandas as pd
import numpy as np
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex

def main():
    print("=== Python 機械学習実装 ===")
    
    # データディレクトリの設定
    data_dir = os.environ.get('DATA_DIR', 'data')
    
    # データ読み込み
    drinks_df = pd.read_csv(f'{data_dir}/drinks.csv', comment='#')
    users_df = pd.read_csv(f'{data_dir}/users.csv', comment='#')
    interactions_df = pd.read_csv(f'{data_dir}/interactions.csv', comment='#')
    
    print(f"データ読み込み完了: ユーザー{len(users_df)}人, 飲み物{len(drinks_df)}種類, インタラクション{len(interactions_df)}件")
    
    # ベクトル化処理
    start_time = time.time()
    drinks_df['ingredients_list'] = drinks_df['ingredients'].str.split('|')
    ingredients_texts = drinks_df['ingredients_list'].apply(lambda x: ' '.join(x))
    
    vectorizer = TfidfVectorizer(min_df=1, max_df=0.7, analyzer='char', ngram_range=(2, 4)) 
    tfidf_matrix = vectorizer.fit_transform(ingredients_texts)
    vectorization_time = time.time() - start_time
    
    print(f"ベクトル化完了: {vectorization_time:.6f}秒")
    
    # 類似度計算
    start_time = time.time()
    similarities = cosine_similarity(tfidf_matrix)
    similarity_time = time.time() - start_time
    
    print(f"類似度計算完了: {similarity_time:.6f}秒")
    
    # Annoy検索
    start_time = time.time()
    vector_dim = tfidf_matrix.shape[1]
    index = AnnoyIndex(vector_dim, 'angular')
    
    for i in range(tfidf_matrix.shape[0]):
        vector = tfidf_matrix[i].toarray().flatten()
        index.add_item(i, vector)
    
    index.build(n_trees=10)
    build_time = time.time() - start_time
    
    # 検索テスト
    start_time = time.time()
    neighbors = index.get_nns_by_item(0, 6)  # lemon_sour (自分を含めて6件)
    search_time = time.time() - start_time
    
    print(f"Annoy検索完了: {search_time:.6f}秒")
    
    # 検索結果表示
    print("\n=== 類似飲み物検索結果 ===")
    target_drink = drinks_df.iloc[0]['name']
    print(f"検索対象: {target_drink}")
    
    for i, neighbor_idx in enumerate(neighbors[1:6], 1):  # 上位5位（自分を除く）
        drink_name = drinks_df.iloc[neighbor_idx]['name']
        category = drinks_df.iloc[neighbor_idx]['category']
        similarity = similarities[0][neighbor_idx]
        print(f"{i}. {drink_name} ({category}) - 類似度: {similarity:.4f}")
    
    # レモンサワーに似た飲み物の類似度を表示
    print("\n=== レモンサワーに似た飲み物の類似度 ===")
    lemon_target_drinks = ['highball_lemon', 'lemon_mojito', 'lemon_juice', 'orange_mojito', 'kahlua_milk']
    for drink_name in lemon_target_drinks:
        drink_idx = drinks_df[drinks_df['name'] == drink_name].index
        if len(drink_idx) > 0:
            idx = drink_idx[0]
            similarity = similarities[0][idx]  # lemon_sourとの類似度
            category = drinks_df.iloc[idx]['category']
            print(f"{drink_name} ({category}) - 類似度: {similarity:.4f}")
    
    # ピーチサワーに似た飲み物の類似度を表示
    print("\n=== ピーチサワーに似た飲み物の類似度 ===")
    peach_sour_idx = drinks_df[drinks_df['name'] == 'peach_sour'].index
    if len(peach_sour_idx) > 0:
        peach_sour_idx = peach_sour_idx[0]
        peach_target_drinks = ['peach_highball', 'peach_mojito', 'peach_juice', 'peach_oolong', 'kahlua_milk']
        for drink_name in peach_target_drinks:
            drink_idx = drinks_df[drinks_df['name'] == drink_name].index
            if len(drink_idx) > 0:
                idx = drink_idx[0]
                similarity = similarities[peach_sour_idx][idx]  # peach_sourとの類似度
                category = drinks_df.iloc[idx]['category']
                print(f"{drink_name} ({category}) - 類似度: {similarity:.4f}")
    else:
        print("ピーチサワーが見つかりません")
    
    # Vim使いランキング
    print("\n=== Vim使いランキング ===")
    vim_users = users_df[users_df['uses_vim'] == 1]['user_id'].tolist()
    vim_interactions = interactions_df[interactions_df['user_id'].isin(vim_users)]
    
    drink_counts = vim_interactions['item_id'].value_counts()
    for i, (drink_id, count) in enumerate(drink_counts.head(3).items(), 1):
        drink_name = drinks_df[drinks_df['drink_id'] == drink_id]['name'].iloc[0]
        category = drinks_df[drinks_df['drink_id'] == drink_id]['category'].iloc[0]
        print(f"{i}. {drink_name} ({category}) - {count}回")
    
    total_time = vectorization_time + similarity_time + search_time
    print(f"\n総処理時間: {total_time:.6f}秒")
    
    # 詳細な性能測定結果
    print(f"\n=== 詳細性能測定結果 ===")
    print(f"Python結果:")
    print(f"ベクトル化完了: {vectorization_time:.6f}秒")
    print(f"類似度計算完了: {similarity_time:.6f}秒")
    print(f"Annoy検索完了: {search_time:.6f}秒")
    print(f"総処理時間: {total_time:.6f}秒")
    
    # 性能測定結果の表形式出力
    print("\n=== 性能測定結果 ===")
    print("| データ量 | Python | Ruby | 性能差 |")
    print("|----------|--------|------|--------|")
    print(f"| {len(drinks_df)}件 | {total_time:.6f}秒 | - | - |")

if __name__ == "__main__":
    main()
