# CSV 読み込み問題の解決

## ディレクトリ構造とパスの問題

### ホスト側の構造

```
/Users/workspace/personal_dev/annoy_RunnestHub/  ← プロジェクトルート
├── data/                    # CSVファイル（ここにアクセスしたい）
│   ├── drinks.csv
│   ├── users.csv
│   └── interactions.csv
├── python/                  # Pythonアプリ
│   ├── main.py             # ここから ../data/ にアクセスしようとした
│   ├── requirements.txt
│   └── Dockerfile
└── ruby/
    ├── main.rb
    └── Dockerfile
```

### コンテナ内の構造（修正前）

```
/app/                        ← Dockerコンテナ内
├── main.py                 # ../data/ は存在しない！
├── requirements.txt
└── Dockerfile
```

### コンテナ内の構造（修正後）

```
/app/                        ← Dockerコンテナ内
├── main.py                 # data/ でアクセス可能
├── requirements.txt
├── data/                   ← 追加された！
│   ├── drinks.csv
│   ├── users.csv
│   └── interactions.csv
└── Dockerfile
```

## 発生したエラー

### エラー 1: FileNotFoundError

```
FileNotFoundError: [Errno 2] No such file or directory: '../data/drinks.csv'
```

**原因**: Docker コンテナ内に data ディレクトリがコピーされていない
**解決**: Dockerfile に`COPY data /app/data`を追加

### エラー 2: Docker ビルドエラー

```
ERROR: failed to solve: failed to compute cache key: "/data": not found
```

**原因**: プロジェクトルートからビルドする必要がある
**解決**: `docker build -f Dockerfile -t python-app .`

#### `-f`オプションとビルドコンテキストについて

- プロジェクトルートからビルドする際、`Dockerfile`の場所を指定する必要がある
- `-f Dockerfile`: 現在のディレクトリ（python）の Dockerfile を指定
- `..`: ビルドコンテキストを親ディレクトリ（プロジェクトルート）に指定
- これにより`python/`と`data/`の両方にアクセス可能

### エラー 3: CSV 読み込みエラー

```
drinks.csvのshape: (11, 1)
drinks.csvのcolumns: ['# drinks.csv']
```

**原因**: CSV ファイルのコメント行がヘッダーとして認識される
**解決**: `pd.read_csv(path, comment='#')`でコメント行を無視

## 修正内容

1. **Dockerfile**: データディレクトリをコピー、ビルドコンテキスト用にパスを調整
2. **main.py**: パスを`data`に変更、`comment='#'`を追加
3. **実行方法**: `python`ディレクトリからビルドコンテキストを親ディレクトリに指定
