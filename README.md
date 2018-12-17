# Text_classification
This project aims to classify wine on the basis of wine tasting reviews with some text analysis and modeling. The data set is publicly available on [Kaggle](https://www.kaggle.com/zynicide/wine-reviews) and can also be found in the **Data** folder. There are in total around 130K records in the data set. The summary of the two columns are presented below.

* **description**: description of the taster
* **variety**: The type of grapes used to make the wine (eg: Pinot Noir)

We are going to use the **description** column as the input and predict the varieties of the wine from the labels in **variety** column.


To predict the wine variety given its review, you need to first download the pre-trained models (SVM model and the tokenizer model) from the **Models** folder into your project directory. In order to start the Flask server, run 

```python
python run_classifier.py
```
and then by post request to the server, you could get the predicted wine variety given its review. For instance,

```python
curl -d '{"description":"This wine's stone fruit and green apple aromas are lightly briny and focused. It features a tight, centered palate with vivid orange, lime and green melon flavors. The finish is compact, minerally and driven by energy."}' -H "Content-Type: application/json" -X POST http://0.0.0.0:5252/classifier
```
