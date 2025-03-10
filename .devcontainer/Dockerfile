# ベースイメージを指定
FROM python:3.11-slim

# 作業ディレクトリを作成・指定
WORKDIR /workspace

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pythonの環境を整える
RUN pip install --upgrade pip

# requirements.txtをコピーしてインストール
COPY requirements.txt .
RUN pip install -r requirements.txt

# dev-requirements.txtをコピーしてインストール（開発用）
COPY dev-requirements.txt .
RUN pip install -r dev-requirements.txt

# コンテナ起動時のデフォルトコマンド
CMD ["sleep", "infinity"]