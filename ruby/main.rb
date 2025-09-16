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
      text = "#{drink['name']} #{ingredients.join(' ')}"
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

    @logger.info("=== 結合データサンプル ===")
    joined_data.first(3).each_with_index do |record, index|
      @logger.info("結合データ #{index + 1}:")
      @logger.info("  ユーザー情報: #{record['user_info']}")
      @logger.info("  ドリンク情報: #{record['drink_info']}")
      @logger.info("  インタラクション: #{record}")
    end
    
    @logger.info("=== 分析完了 ===")
  end
end

if __FILE__ == $0
  analyzer = RunnestHubAnalyzer.new
  analyzer.run_analysis
end
