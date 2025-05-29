HF_TOKEN=...
DIR=$PWD

wget -P . --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/CoinGradingDL/CoinGradingDataset/resolve/main/1k-coins-dataset-no-pr.tar.gz
tar -xzvf 1k-coins-dataset-no-pr.tar.gz

cd $DIR

wget -P . --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/CoinGradingDL/CoinGradingDataset/resolve/main/1k-coins-dataset-no-pr.csv
