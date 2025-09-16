require 'csv'
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
      CSV.foreach(file_path, headers: true) do |row|
        data << row.to_h
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

  def run_analysis
    @logger.info("=== RunnestHub データ分析開始 ===")

    users_data = load_csv_data('../data/users.csv')
    drinks_data = load_csv_data('../data/drinks.csv')
    interactions_data = load_csv_data('../data/interactions.csv')

    user_analysis = analyze_users(users_data)
    drink_analysis = analyze_drinks(drinks_data)
    interaction_analysis = analyze_interactions(interactions_data)

    @logger.info("=== 分析結果 ===")
    @logger.info("ユーザー分析: #{user_analysis}")
    @logger.info("ドリンク分析: #{drink_analysis}")
    @logger.info("インタラクション分析: #{interaction_analysis}")
    
    @logger.info("=== 分析完了 ===")
  end
end

if __FILE__ == $0
  analyzer = RunnestHubAnalyzer.new
  analyzer.run_analysis
end
