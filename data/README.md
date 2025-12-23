## To acquire the data

```
export KAGGLE_API_TOKEN=your_api_key
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge

# Unzip data

 unzip ../jigsaw-toxic-comment-classification-challenge.zip -d ../data/raw
 unzip ../data/raw/test.csv.zip -d ../data/raw
 unzip ../data/raw/test_labels.csv.zip -d ../data/raw
 unzip ../data/raw/train.csv.zip -d ../data/raw

# Remove unwanted files

 rm ../jigsaw-toxic-comment-classification-challenge.zip
 rm ../data/raw/sample_submission.csv.zip
 rm ../data/raw/test.csv.zip
 rm ../data/raw/test_labels.csv.zip
 rm ../data/raw/train.csv.zip
```


