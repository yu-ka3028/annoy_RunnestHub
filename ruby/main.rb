require 'csv'
require 'matrix'
require 'tf-idf-similarity'
require 'annoy'
require 'logger'

class RunnestHubAnalyzer
  def initialize
    @logger = Logger.new(STDOUT)
    @logger.level = Logger::INFO
    @logger.formatter = proc do |severity, datetime, progname, msg|
      "#{datetime.strftime('%Y-%m-%d %H:%M:%S')} [#{severity}] #{msg}\n"
    end
  end

  def load_csv_data(file_path)
    @logger.info("CSVファイルを読み込み中: #{file_path}")
    
    begin
      data = []
      CSV.foreach(file_path, headers: true, skip_lines: /^#/) do |row|
        row_data = row.to_h

        if row_data['ingredients']
          row_data['ingredients'] = row_data['ingredients'].split('|')
        end
        
        data << row_data
      end
      @logger.info("CSV読み込み完了: #{data.length}行のデータを読み込みました")
      data
    rescue => e
      @logger.error("CSV読み込みエラー: #{e.message}")
      []
    end
  end

  def analyze_users(users_data)
    @logger.info("ユーザー属性の分析を開始")
    
    return {} if users_data.empty?

    gender_stats = users_data.group_by { |user| user['gender'] }
                            .transform_values(&:length)
    ages = users_data.map { |user| user['age'].to_i }.compact
    age_stats = {
      count: ages.length,
      min: ages.min,
      max: ages.max,
      average: ages.sum.to_f / ages.length
    }
    
    {
      total_users: users_data.length,
      gender_distribution: gender_stats,
      age_statistics: age_stats
    }
  end

  def analyze_drinks(drinks_data)
    @logger.info("ドリンクデータの分析を開始")
    
    return {} if drinks_data.empty?

    category_stats = drinks_data.group_by { |drink| drink['category'] }
                                .transform_values(&:length)

    prices = drinks_data.map { |drink| drink['price'].to_f }.compact
    price_stats = {
      count: prices.length,
      min: prices.min,
      max: prices.max,
      average: prices.sum / prices.length
    }
    
    {
      total_drinks: drinks_data.length,
      category_distribution: category_stats,
      price_statistics: price_stats
    }
  end

  def analyze_interactions(interactions_data)
    @logger.info("インタラクションデータの分析を開始")
    
    return {} if interactions_data.empty?

    user_interactions = interactions_data.group_by { |interaction| interaction['user_id'] }
                                        .transform_values(&:length)

    drink_interactions = interactions_data.group_by { |interaction| interaction['drink_id'] }
                                          .transform_values(&:length)
    
    {
      total_interactions: interactions_data.length,
      unique_users: user_interactions.keys.length,
      unique_drinks: drink_interactions.keys.length,
      avg_interactions_per_user: interactions_data.length.to_f / user_interactions.keys.length,
      avg_interactions_per_drink: interactions_data.length.to_f / drink_interactions.keys.length
    }
  end

  def vectorize_drinks(drinks_data)
    @logger.info("ドリンクのベクトル化を開始")
    
    return {} if drinks_data.empty?

    drink_texts = drinks_data.map do |drink|
      ingredients = drink['ingredients'] || []

      text = ingredients.join(' ')
      {
        drink_id: drink['drink_id'],
        name: drink['name'],
        text: text
      }
    end

    # 独自のTF-IDF実装
    tfidf_result = calculate_tfidf(drink_texts)
    
    @logger.info("ベクトル化完了: #{tfidf_result[:drink_vectors].length}個のドリンクをベクトル化しました")
    
    tfidf_result
  end

  def calculate_tfidf(drink_texts)
    # 全ドキュメントから語彙を抽出
    vocabulary = Set.new
    documents = drink_texts.map do |drink|
      words = drink[:text].downcase.split(/\s+/)
      vocabulary.merge(words)
      words
    end
    
    vocabulary = vocabulary.to_a.sort
    @logger.info("語彙数: #{vocabulary.length}")
    
    # TF（Term Frequency）を計算
    tf_matrix = documents.map do |words|
      word_count = words.tally
      vocabulary.map { |word| word_count[word] || 0 }
    end
    
    # IDF（Inverse Document Frequency）を計算
    idf_scores = vocabulary.map do |word|
      docs_containing_word = documents.count { |words| words.include?(word) }
      Math.log(documents.length.to_f / docs_containing_word)
    end
    
    # TF-IDFを計算
    drink_vectors = {}
    drink_texts.each_with_index do |drink, index|
      tfidf_vector = tf_matrix[index].zip(idf_scores).map { |tf, idf| tf * idf }
      
      drink_vectors[drink[:drink_id]] = {
        name: drink[:name],
        vector: tfidf_vector,
        text: drink[:text]
      }
    end
    
    {
      drink_vectors: drink_vectors,
      vocabulary: vocabulary,
      tf_matrix: tf_matrix,
      idf_scores: idf_scores
    }
  end

  def find_similar_drinks(drink_vectors, target_drink_id, top_n = 5)
    @logger.info("類似ドリンクの検索を開始: ドリンクID #{target_drink_id}")
    
    return [] unless drink_vectors[target_drink_id]

    target_vector = drink_vectors[target_drink_id][:vector]
    similarities = []

    drink_vectors.each do |drink_id, data|
      next if drink_id == target_drink_id
      
      similarity = cosine_similarity(target_vector, data[:vector])
      similarities << {
        drink_id: drink_id,
        name: data[:name],
        similarity: similarity
      }
    end

    similarities.sort_by { |s| -s[:similarity] }.first(top_n)
  end

  def cosine_similarity(vector_a, vector_b)
    return 0.0 if vector_a.empty? || vector_b.empty?
    
    dot_product = vector_a.zip(vector_b).map { |a, b| a * b }.sum
    magnitude_a = Math.sqrt(vector_a.map { |x| x**2 }.sum)
    magnitude_b = Math.sqrt(vector_b.map { |x| x**2 }.sum)
    
    return 0.0 if magnitude_a == 0 || magnitude_b == 0
    
    dot_product / (magnitude_a * magnitude_b)
  end

  def build_annoy_index(drink_vectors, dimensions)
    @logger.info("Annoyインデックスの構築を開始")
    
    # Annoyインデックスを作成（コサイン類似度用）
    index = Annoy::AnnoyIndexAngular.new(dimensions)
    
    drink_vectors.each_with_index do |(drink_id, data), idx|
      vector = data[:vector]
      normalized_vector = normalize_vector(vector)
      index.add_item(idx, normalized_vector)
    end
    

    index.build(10, -1)
    
    @logger.info("Annoyインデックス構築完了: #{drink_vectors.length}個のベクトルをインデックス化")
    
    index
  end

  def normalize_vector(vector)
    magnitude = Math.sqrt(vector.map { |x| x**2 }.sum)
    return vector.map { |x| 0.0 } if magnitude == 0
    
    vector.map { |x| x / magnitude }
  end

  def find_annoy_neighbors(index, drink_vectors, target_drink_id, k = 5)
    @logger.info("Annoy近似K近傍検索を開始: ドリンクID #{target_drink_id}")
    
    return [] unless drink_vectors[target_drink_id]
    
    target_vector = normalize_vector(drink_vectors[target_drink_id][:vector])
    neighbors = index.get_nns_by_vector(target_vector, k + 1, -1, true)
    neighbor_indices = neighbors[0]
    distances = neighbors[1]
    
    results = []
    drink_ids = drink_vectors.keys
    
    neighbor_indices.each_with_index do |neighbor_idx, i|

      next if drink_ids[neighbor_idx] == target_drink_id
      
      drink_id = drink_ids[neighbor_idx]
      data = drink_vectors[drink_id]
      
      similarity = 1.0 - distances[i]
      
      results << {
        drink_id: drink_id,
        name: data[:name],
        similarity: similarity,
        distance: distances[i]
      }
    end
    
    @logger.info("Annoy検索完了: #{results.length}個の近傍を発見")
    results
  end

  def compare_search_methods(drink_vectors, target_drink_id, k = 5)
    @logger.info("=== 検索手法の比較 ===")
    
    linear_results = find_similar_drinks(drink_vectors, target_drink_id, k)
    
    dimensions = drink_vectors.values.first[:vector].length
    annoy_index = build_annoy_index(drink_vectors, dimensions)
    annoy_results = find_annoy_neighbors(annoy_index, drink_vectors, target_drink_id, k)
    
    @logger.info("=== 検索結果比較 ===")
    @logger.info("ターゲットドリンク: #{drink_vectors[target_drink_id][:name]}")
    
    @logger.info("--- 線形検索結果 ---")
    linear_results.each_with_index do |result, i|
      @logger.info("#{i + 1}. #{result[:name]} (類似度: #{result[:similarity].round(4)})")
    end
    
    @logger.info("--- Annoy近似検索結果 ---")
    annoy_results.each_with_index do |result, i|
      @logger.info("#{i + 1}. #{result[:name]} (類似度: #{result[:similarity].round(4)}, 距離: #{result[:distance].round(4)})")
    end
    
    linear_names = linear_results.map { |r| r[:name] }
    annoy_names = annoy_results.map { |r| r[:name] }
    
    common_results = linear_names & annoy_names
    @logger.info("--- 比較結果 ---")
    @logger.info("共通結果数: #{common_results.length}/#{k}")
    @logger.info("共通結果: #{common_results}")
    
    {
      linear_search: linear_results,
      annoy_search: annoy_results,
      common_results: common_results,
      common_count: common_results.length
    }
  end

  def join_data(users_data, drinks_data, interactions_data)
    @logger.info("データの結合処理を開始")

    users_hash = users_data.each_with_object({}) do |user, hash|
      user_id = user['user_id']
      hash[user_id] = user
      hash[user_id.to_i] = user
    end

    drinks_hash = drinks_data.each_with_object({}) do |drink, hash|
      drink_id = drink['drink_id']
      hash[drink_id] = drink
      hash[drink_id.to_i] = drink
    end

    joined_data = interactions_data.map do |interaction|
      user_id = interaction['user_id']
      drink_id = interaction['item_id']
      
      joined_record = interaction.dup
      joined_record['user_info'] = users_hash[user_id] || users_hash[user_id.to_i] || {}
      joined_record['drink_info'] = drinks_hash[drink_id] || drinks_hash[drink_id.to_i] || {}
      
      joined_record
    end
    
    @logger.info("データ結合完了: #{joined_data.length}件の結合データを生成しました")
    joined_data
  end

  def analyze_vim_drinker_ranking(users_data, interactions_data, drinks_data)
    @logger.info("Vim使いがよく飲むお酒ランキングの分析を開始")
    
    vim_users = users_data.select { |user| user['uses_vim'] == '1' }
    vim_user_ids = vim_users.map { |user| user['user_id'] }
    
    @logger.info("Vim使いのユーザー数: #{vim_user_ids.length}")
    @logger.info("Vim使いのユーザーID: #{vim_user_ids}")
    
    vim_interactions = interactions_data.select do |interaction|
      vim_user_ids.include?(interaction['user_id'])
    end
    
    @logger.info("Vim使いのインタラクション数: #{vim_interactions.length}")
    
    drink_counts = vim_interactions.group_by { |interaction| interaction['item_id'] }
                                  .transform_values(&:length)
    
    drinks_hash = drinks_data.each_with_object({}) do |drink, hash|
      drink_id = drink['drink_id']
      hash[drink_id] = drink
      hash[drink_id.to_i] = drink
    end
    
    ranking = drink_counts.map do |drink_id, count|
      drink_info = drinks_hash[drink_id] || drinks_hash[drink_id.to_i] || {}
      {
        drink_id: drink_id,
        name: drink_info['name'] || "Unknown",
        category: drink_info['category'] || "Unknown",
        count: count
      }
    end
    
    ranking.sort_by { |item| -item[:count] }
  end

  def analyze_attribute_ranking(users_data, interactions_data, drinks_data, filters = {})
    @logger.info("属性別ランキングの分析を開始")
    @logger.info("フィルター条件: #{filters}")
    
    # フィルター条件に基づいてユーザーを選択
    filtered_users = users_data.select do |user|
      filters.all? do |attribute, value|
        if value.is_a?(Array)
          value.include?(user[attribute])
        else
          user[attribute] == value
        end
      end
    end
    
    filtered_user_ids = filtered_users.map { |user| user['user_id'] }
    
    @logger.info("フィルター後のユーザー数: #{filtered_user_ids.length}")
    @logger.info("フィルター後のユーザーID: #{filtered_user_ids}")
    
    # フィルターされたユーザーのインタラクションを取得
    filtered_interactions = interactions_data.select do |interaction|
      filtered_user_ids.include?(interaction['user_id'])
    end
    
    @logger.info("フィルター後のインタラクション数: #{filtered_interactions.length}")
    
    # ドリンクIDごとのカウントを集計
    drink_counts = filtered_interactions.group_by { |interaction| interaction['item_id'] }
                                      .transform_values(&:length)
    
    # ドリンク情報と結合してランキングを作成
    drinks_hash = drinks_data.each_with_object({}) do |drink, hash|
      drink_id = drink['drink_id']
      hash[drink_id] = drink
      hash[drink_id.to_i] = drink
    end
    
    ranking = drink_counts.map do |drink_id, count|
      drink_info = drinks_hash[drink_id] || drinks_hash[drink_id.to_i] || {}
      {
        drink_id: drink_id,
        name: drink_info['name'] || "Unknown",
        category: drink_info['category'] || "Unknown",
        count: count
      }
    end
    
    ranking.sort_by { |item| -item[:count] }
  end

  def analyze_multi_attribute_ranking(users_data, interactions_data, drinks_data, attribute_combinations)
    @logger.info("複数属性組み合わせランキングの分析を開始")
    
    results = {}
    
    attribute_combinations.each do |combination_name, filters|
      @logger.info("=== #{combination_name} ===")
      ranking = analyze_attribute_ranking(users_data, interactions_data, drinks_data, filters)
      results[combination_name] = ranking
    end
    
    results
  end

  def display_ranking(title, ranking)
    @logger.info("--- #{title}ランキング ---")
    
    if ranking.empty?
      @logger.info("#{title}の飲み物データが見つかりませんでした")
    else
      @logger.info("ランキング結果:")
      ranking.each_with_index do |item, index|
        @logger.info("#{index + 1}. #{item[:name]} (#{item[:category]}) - #{item[:count]}回")
      end
    end
    @logger.info("")
  end

  def display_attribute_distribution(users_data)
    @logger.info("=== ユーザー属性分布 ===")
    
    gender_dist = users_data.group_by { |user| user['gender'] }
                            .transform_values(&:length)
    @logger.info("性別分布:")
    gender_dist.each do |gender, count|
      percentage = (count.to_f / users_data.length * 100).round(1)
      @logger.info("  #{gender}: #{count}人 (#{percentage}%)")
    end
    
    lang_dist = users_data.group_by { |user| user['favorite_lang'] }
                          .transform_values(&:length)
    @logger.info("プログラミング言語分布:")
    lang_dist.sort_by { |_, count| -count }.each do |lang, count|
      percentage = (count.to_f / users_data.length * 100).round(1)
      @logger.info("  #{lang}: #{count}人 (#{percentage}%)")
    end
    
    os_dist = users_data.group_by { |user| user['os'] }
                        .transform_values(&:length)
    @logger.info("OS分布:")
    os_dist.sort_by { |_, count| -count }.each do |os, count|
      percentage = (count.to_f / users_data.length * 100).round(1)
      @logger.info("  #{os}: #{count}人 (#{percentage}%)")
    end
    
    editor_dist = users_data.group_by { |user| user['editor'] }
                            .transform_values(&:length)
    @logger.info("エディタ分布:")
    editor_dist.sort_by { |_, count| -count }.each do |editor, count|
      percentage = (count.to_f / users_data.length * 100).round(1)
      @logger.info("  #{editor}: #{count}人 (#{percentage}%)")
    end
    
    vim_users = users_data.count { |user| user['uses_vim'] == '1' }
    vim_percentage = (vim_users.to_f / users_data.length * 100).round(1)
    @logger.info("Vim使用率: #{vim_users}/#{users_data.length}人 (#{vim_percentage}%)")
    
    night_owl_users = users_data.count { |user| user['night_owl'] == '1' }
    night_owl_percentage = (night_owl_users.to_f / users_data.length * 100).round(1)
    @logger.info("夜型率: #{night_owl_users}/#{users_data.length}人 (#{night_owl_percentage}%)")
    
    extroversion_dist = users_data.group_by { |user| user['extroversion_tag'] }
                                  .transform_values(&:length)
    @logger.info("外向性分布:")
    extroversion_dist.sort_by { |_, count| -count }.each do |type, count|
      percentage = (count.to_f / users_data.length * 100).round(1)
      @logger.info("  #{type}: #{count}人 (#{percentage}%)")
    end
    
    @logger.info("")
  end

  def display_drink_category_distribution(drinks_data)
    @logger.info("=== 飲み物カテゴリ分布 ===")
    
    category_dist = drinks_data.group_by { |drink| drink['category'] }
                               .transform_values(&:length)
    
    category_dist.sort_by { |_, count| -count }.each do |category, count|
      percentage = (count.to_f / drinks_data.length * 100).round(1)
      @logger.info("#{category}: #{count}種類 (#{percentage}%)")
    end
    
    @logger.info("")
  end

  def calculate_basic_statistics(users_data, drinks_data, interactions_data)
    @logger.info("=== 基本統計量 ===")
    
    ages = users_data.map { |user| user['age'].to_i }.compact
    if ages.any?
      ages_sorted = ages.sort
      @logger.info("年齢統計:")
      @logger.info("  平均: #{ages.sum.to_f / ages.length}歳")
      @logger.info("  中央値: #{ages_sorted[ages_sorted.length / 2]}歳")
      @logger.info("  最小: #{ages.min}歳")
      @logger.info("  最大: #{ages.max}歳")
      @logger.info("  標準偏差: #{calculate_standard_deviation(ages)}")
    end
    
    coding_hours = users_data.map { |user| user['coding_hours_per_day'].to_i }.compact
    if coding_hours.any?
      coding_hours_sorted = coding_hours.sort
      @logger.info("1日のコーディング時間統計:")
      @logger.info("  平均: #{coding_hours.sum.to_f / coding_hours.length}時間")
      @logger.info("  中央値: #{coding_hours_sorted[coding_hours_sorted.length / 2]}時間")
      @logger.info("  最小: #{coding_hours.min}時間")
      @logger.info("  最大: #{coding_hours.max}時間")
    end

    prices = drinks_data.map { |drink| drink['price'].to_f }.compact
    if prices.any?
      prices_sorted = prices.sort
      @logger.info("ドリンク価格統計:")
      @logger.info("  平均: #{prices.sum / prices.length}円")
      @logger.info("  中央値: #{prices_sorted[prices_sorted.length / 2]}円")
      @logger.info("  最小: #{prices.min}円")
      @logger.info("  最大: #{prices.max}円")
    end
    
    abvs = drinks_data.map { |drink| drink['abv'].to_f }.compact
    if abvs.any?
      abvs_sorted = abvs.sort
      @logger.info("アルコール度数統計:")
      @logger.info("  平均: #{abvs.sum / abvs.length}%")
      @logger.info("  中央値: #{abvs_sorted[abvs_sorted.length / 2]}%")
      @logger.info("  最小: #{abvs.min}%")
      @logger.info("  最大: #{abvs.max}%")
    end
    
    user_interaction_counts = interactions_data.group_by { |i| i['user_id'] }
                                              .transform_values(&:length)
                                              .values
    if user_interaction_counts.any?
      interaction_counts_sorted = user_interaction_counts.sort
      @logger.info("ユーザーあたりのインタラクション数統計:")
      @logger.info("  平均: #{user_interaction_counts.sum.to_f / user_interaction_counts.length}回")
      @logger.info("  中央値: #{interaction_counts_sorted[interaction_counts_sorted.length / 2]}回")
      @logger.info("  最小: #{user_interaction_counts.min}回")
      @logger.info("  最大: #{user_interaction_counts.max}回")
    end
    
    @logger.info("")
  end

  def calculate_standard_deviation(values)
    return 0.0 if values.empty?
    
    mean = values.sum.to_f / values.length
    variance = values.map { |v| (v - mean) ** 2 }.sum / values.length
    Math.sqrt(variance).round(2)
  end

  def run_analysis
    @logger.info("=== RunnestHub データ分析開始 ===")

    users_data = load_csv_data('data/users.csv')
    drinks_data = load_csv_data('data/drinks.csv')
    interactions_data = load_csv_data('data/interactions.csv')

    joined_data = join_data(users_data, drinks_data, interactions_data)

    user_analysis = analyze_users(users_data)
    drink_analysis = analyze_drinks(drinks_data)
    interaction_analysis = analyze_interactions(interactions_data)

    vectorization_result = vectorize_drinks(drinks_data)

    @logger.info("=== 分析結果 ===")
    @logger.info("ユーザー分析: #{user_analysis}")
    @logger.info("ドリンク分析: #{drink_analysis}")
    @logger.info("インタラクション分析: #{interaction_analysis}")

    # 統計情報の表示
    display_attribute_distribution(users_data)
    display_drink_category_distribution(drinks_data)
    calculate_basic_statistics(users_data, drinks_data, interactions_data)

    @logger.info("=== ベクトル化結果 ===")
    if vectorization_result[:drink_vectors]
      @logger.info("語彙数: #{vectorization_result[:vocabulary].length}")
      @logger.info("ベクトル化されたドリンク数: #{vectorization_result[:drink_vectors].length}")
      
      vectorization_result[:drink_vectors].each do |drink_id, data|
        @logger.info("ドリンクID #{drink_id}: #{data[:name]}")
        @logger.info("  テキスト: #{data[:text]}")
        @logger.info("  ベクトル次元数: #{data[:vector].length}")
        @logger.info("  ベクトル（最初の10要素）: #{data[:vector].first(10)}")
      end
    end

    @logger.info("=== 類似ドリンク検索デモ ===")
    if vectorization_result[:drink_vectors] && !vectorization_result[:drink_vectors].empty?

      first_drink_id = vectorization_result[:drink_vectors].keys.first
      similar_drinks = find_similar_drinks(vectorization_result[:drink_vectors], first_drink_id, 3)
      
      @logger.info("ドリンクID #{first_drink_id} の類似ドリンク:")
      similar_drinks.each do |similar|
        @logger.info("  - #{similar[:name]} (類似度: #{similar[:similarity].round(4)})")
      end
    end

    @logger.info("=== 近傍検索の比較デモ ===")
    if vectorization_result[:drink_vectors] && !vectorization_result[:drink_vectors].empty?

      test_drinks = vectorization_result[:drink_vectors].keys.first(3)
      
      test_drinks.each do |drink_id|
        @logger.info("\n--- ドリンクID #{drink_id} での比較 ---")
        comparison_result = compare_search_methods(vectorization_result[:drink_vectors], drink_id, 3)
        
        @logger.info("検索精度: #{comparison_result[:common_count]}/3 件が一致")
      end
    end

    @logger.info("=== 結合データサンプル ===")
    joined_data.first(3).each_with_index do |record, index|
      @logger.info("結合データ #{index + 1}:")
      @logger.info("  ユーザー情報: #{record['user_info']}")
      @logger.info("  ドリンク情報: #{record['drink_info']}")
      @logger.info("  インタラクション: #{record}")
    end

    @logger.info("=== Vim使いがよく飲むお酒ランキング ===")
    vim_ranking = analyze_vim_drinker_ranking(users_data, interactions_data, drinks_data)
    display_ranking("Vim使い", vim_ranking)

    @logger.info("=== 属性別ランキング分析 ===")
    
    single_attribute_examples = [
      { name: "男性がよく飲むお酒", filters: { 'gender' => 'male' } },
      { name: "女性がよく飲むお酒", filters: { 'gender' => 'female' } },
      { name: "Python使いがよく飲むお酒", filters: { 'favorite_lang' => 'Python' } },
      { name: "Ruby使いがよく飲むお酒", filters: { 'favorite_lang' => 'Ruby' } },
      { name: "macユーザーがよく飲むお酒", filters: { 'os' => 'mac' } },
      { name: "夜型の人がよく飲むお酒", filters: { 'night_owl' => '1' } }
    ]
    
    single_attribute_examples.each do |example|
      ranking = analyze_attribute_ranking(users_data, interactions_data, drinks_data, example[:filters])
      display_ranking(example[:name], ranking)
    end
    
    multi_attribute_examples = {
      "男性のVim使い" => { 'gender' => 'male', 'uses_vim' => '1' },
      "女性のPython使い" => { 'gender' => 'female', 'favorite_lang' => 'Python' },
      "macの夜型ユーザー" => { 'os' => 'mac', 'night_owl' => '1' },
      "内向的なVim使い" => { 'extroversion_tag' => 'introvert', 'uses_vim' => '1' }
    }
    
    multi_results = analyze_multi_attribute_ranking(users_data, interactions_data, drinks_data, multi_attribute_examples)
    multi_results.each do |combination_name, ranking|
      display_ranking(combination_name, ranking)
    end
    
    @logger.info("=== 分析完了 ===")
  end
end

if __FILE__ == $0
  analyzer = RunnestHubAnalyzer.new
  analyzer.run_analysis
end
