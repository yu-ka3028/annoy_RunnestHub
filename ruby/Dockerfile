FROM ruby:3.1-slim

# ビルドツールをインストール（annoyのコンパイルに必要）
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ruby/Gemfile .
RUN bundle install
COPY ruby/ /app
COPY data /app/data
CMD ["ruby", "main.rb"]