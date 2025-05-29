HF_TOKEN=...
DIR=$PWD

# Download the tar.gz file
wget -P . --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/CoinGradingDL/CoinGradingDataset/resolve/main/1k-coins-dataset-no-pr.tar.gz

# Unpack the tar.gz file
tar -xzvf 1k-coins-dataset-no-pr.tar.gz

cd $DIR

# Download the CSV file
wget -P . --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/CoinGradingDL/CoinGradingDataset/resolve/main/1k-coins-dataset-no-pr.csv
