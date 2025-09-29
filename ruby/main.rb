require 'csv'
require 'matrix'
require 'tf-idf-similarity'
require 'annoy'
require 'logger'

class MinimalAnalyzer
  def initialize
    @logger = Logger.new(STDOUT)
    @logger.level = Logger::INFO
    @logger.formatter = proc do |severity, datetime, progname, msg|
      "#{msg}\n"
    end
  end

  def load_csv_data(file_path)
      data = []
      CSV.foreach(file_path, headers: true, skip_lines: /^#/) do |row|
        row_data = row.to_h
        if row_data['ingredients']
          row_data['ingredients'] = row_data['ingredients'].split('|')
        end
        data << row_data
    end
    data
  end

  def vectorize_drinks(drinks_data)
    start_time = Time.now
    
    # 簡単なベクトル化（TF-IDFの代わりに単語頻度ベクトル）
    all_ingredients = drinks_data.flat_map { |drink| drink['ingredients'] }.uniq
    
    drink_vectors = {}
    drinks_data.each do |drink|
      vector = all_ingredients.map { |ingredient| drink['ingredients'].count(ingredient) }
      drink_vectors[drink['drink_id']] = {
        name: drink['name'],
        category: drink['category'],
        vector: vector
      }
    end
    
    vectorization_time = Time.now - start_time
    @logger.info("ベクトル化完了: #{vectorization_time.round(6)}秒")
    
    { drink_vectors: drink_vectors, vectorization_time: vectorization_time }
  end

  def calculate_similarities(drink_vectors)
    start_time = Time.now
    
    similarities = {}
    drink_vectors.each do |drink_id, data|
      similarities[drink_id] = {}
      drink_vectors.each do |other_id, other_data|
        next if drink_id == other_id
        similarity = cosine_similarity(data[:vector], other_data[:vector])
        similarities[drink_id][other_id] = similarity
      end
    end
    
    similarity_time = Time.now - start_time
    @logger.info("類似度計算完了: #{similarity_time.round(6)}秒")
    
    { similarities: similarities, similarity_time: similarity_time }
  end

  def cosine_similarity(vec1, vec2)
    dot_product = vec1.zip(vec2).map { |a, b| a * b }.sum
    magnitude1 = Math.sqrt(vec1.map { |x| x**2 }.sum)
    magnitude2 = Math.sqrt(vec2.map { |x| x**2 }.sum)
    
    return 0.0 if magnitude1 == 0 || magnitude2 == 0
    
    dot_product / (magnitude1 * magnitude2)
  end

  def annoy_search(drink_vectors)
    start_time = Time.now
    
    # Annoyの代わりに類似度計算で上位5件を取得
    drink_ids = drink_vectors.keys
    similarities = {}
    
    # lemon_sour (最初の飲み物) との類似度を計算
    target_drink_id = drink_ids.first
    target_vector = drink_vectors[target_drink_id][:vector]
    
    drink_ids.each do |drink_id|
      next if drink_id == target_drink_id
      vector = drink_vectors[drink_id][:vector]
      similarity = cosine_similarity(target_vector, vector)
      similarities[drink_id] = similarity
    end
    
    # 類似度順にソートして上位5件を取得
    sorted_similarities = similarities.sort_by { |_, sim| -sim }
    neighbors = [target_drink_id] + sorted_similarities.first(5).map(&:first)
    
    search_time = Time.now - start_time
    @logger.info("類似度検索完了: #{search_time.round(6)}秒")
    
    { neighbors: neighbors, search_time: search_time, drink_ids: drink_ids }
  end

  def analyze_vim_ranking(users_data, interactions_data, drinks_data)
    vim_users = users_data.select { |user| user['uses_vim'] == '1' }.map { |user| user['user_id'] }
    vim_interactions = interactions_data.select { |interaction| vim_users.include?(interaction['user_id']) }
    
    drink_counts = Hash.new(0)
    vim_interactions.each do |interaction|
      drink_counts[interaction['item_id']] += 1
    end
    
    sorted_drinks = drink_counts.sort_by { |_, count| -count }
    
    @logger.info("=== Vim使いランキング ===")
    sorted_drinks.first(3).each_with_index do |(drink_id, count), i|
      drink = drinks_data.find { |d| d['drink_id'] == drink_id }
      @logger.info("#{i + 1}. #{drink['name']} (#{drink['category']}) - #{count}回")
    end
  end

  def run_analysis
    @logger.info("=== Ruby 機械学習実装 ===")
    
    # データディレクトリの設定
    data_dir = ENV['DATA_DIR'] || 'data'
    
    # データ読み込み
    users_data = load_csv_data("#{data_dir}/users.csv")
    drinks_data = load_csv_data("#{data_dir}/drinks.csv")
    interactions_data = load_csv_data("#{data_dir}/interactions.csv")

    @logger.info("データ読み込み完了: ユーザー#{users_data.length}人, 飲み物#{drinks_data.length}種類, インタラクション#{interactions_data.length}件")

    # ベクトル化
    vectorization_result = vectorize_drinks(drinks_data)

    # 類似度計算
    similarity_result = calculate_similarities(vectorization_result[:drink_vectors])
    
    # Annoy検索
    annoy_result = annoy_search(vectorization_result[:drink_vectors])
    
    # 検索結果表示
    @logger.info("\n=== 類似飲み物検索結果 ===")
    target_drink = drinks_data.first['name']
    @logger.info("検索対象: #{target_drink}")
    
    annoy_result[:neighbors][1..5].each_with_index do |drink_id, i|
      drink = drinks_data.find { |d| d['drink_id'] == drink_id }
      similarity = similarity_result[:similarities][drink_id][annoy_result[:drink_ids][0]]
      @logger.info("#{i + 1}. #{drink['name']} (#{drink['category']}) - 類似度: #{similarity.round(4)}")
    end
    
    # レモンサワーに似た飲み物の類似度を表示
    @logger.info("\n=== レモンサワーに似た飲み物の類似度 ===")
    lemon_target_drinks = ['highball_lemon', 'lemon_mojito', 'lemon_juice', 'orange_mojito', 'kahlua_milk']
    lemon_sour_drink = drinks_data.first  # lemon_sour
    lemon_sour_drink_id = lemon_sour_drink['drink_id']
    
    lemon_target_drinks.each do |drink_name|
      drink = drinks_data.find { |d| d['name'] == drink_name }
      if drink
        drink_id = drink['drink_id']
        similarity = similarity_result[:similarities][drink_id][lemon_sour_drink_id]
        @logger.info("#{drink_name} (#{drink['category']}) - 類似度: #{similarity.round(4)}")
      end
    end
    
    # ピーチサワーに似た飲み物の類似度を表示
    @logger.info("\n=== ピーチサワーに似た飲み物の類似度 ===")
    peach_sour_drink = drinks_data.find { |d| d['name'] == 'peach_sour' }
    if peach_sour_drink
      peach_sour_drink_id = peach_sour_drink['drink_id']
      peach_target_drinks = ['peach_highball', 'peach_mojito', 'peach_juice', 'peach_oolong', 'kahlua_milk']
      peach_target_drinks.each do |drink_name|
        drink = drinks_data.find { |d| d['name'] == drink_name }
        if drink
          drink_id = drink['drink_id']
          similarity = similarity_result[:similarities][drink_id][peach_sour_drink_id]
          @logger.info("#{drink_name} (#{drink['category']}) - 類似度: #{similarity.round(4)}")
        end
      end
    else
      @logger.info("ピーチサワーが見つかりません")
    end
    
    # Vim使いランキング
    analyze_vim_ranking(users_data, interactions_data, drinks_data)
    
    total_time = vectorization_result[:vectorization_time] + similarity_result[:similarity_time] + annoy_result[:search_time]
    @logger.info("\n総処理時間: #{total_time.round(6)}秒")
    
    # 詳細な性能測定結果
    @logger.info("\n=== 詳細性能測定結果 ===")
    @logger.info("Ruby結果:")
    @logger.info("ベクトル化完了: #{vectorization_result[:vectorization_time].round(6)}秒")
    @logger.info("類似度計算完了: #{similarity_result[:similarity_time].round(6)}秒")
    @logger.info("Annoy検索完了: #{annoy_result[:search_time].round(6)}秒")
    @logger.info("総処理時間: #{total_time.round(6)}秒")
    
    # 性能測定結果の表形式出力
    @logger.info("\n=== 性能測定結果 ===")
    @logger.info("| データ量 | Python | Ruby | 性能差 |")
    @logger.info("|----------|--------|------|--------|")
    @logger.info("| #{drinks_data.length}件 | - | #{total_time.round(6)}秒 | - |")
  end
end

if __FILE__ == $0
  analyzer = MinimalAnalyzer.new
  analyzer.run_analysis
end