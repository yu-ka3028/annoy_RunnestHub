FROM ruby:3.1-slim

# ビルドツールをインストール（annoyのコンパイルに必要）
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Gemfile .
RUN bundle install
COPY . /app
CMD ["ruby", "main.rb"]