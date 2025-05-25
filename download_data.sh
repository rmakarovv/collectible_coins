HF_TOKEN=...

DIR=$PWD
cd /
wget  --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/CoinGradingDL/CoinGradingDataset/resolve/main/1k-coins-dataset-no-pr.tar.gz
cd $DIR
wget  --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/CoinGradingDL/CoinGradingDataset/resolve/main/1k-coins-dataset-no-pr.csv

