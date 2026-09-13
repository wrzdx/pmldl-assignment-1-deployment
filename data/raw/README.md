# Iris data

`iris.csv` contains the 150 records exported from `sklearn.datasets.load_iris`:
four measurements in centimetres and one of three species labels.
This is the **flower Iris dataset**, not iris/face photographs or patient smoking data.

Sources:
- [scikit-learn Iris loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
- [UCI Iris dataset](https://archive.ics.uci.edu/dataset/53/iris)

The CSV is included so ordinary pipeline runs need no dataset download.
To recreate it using the pinned scikit-learn release:

```bash
python -m iris_ml.datasets.export_raw
```

The raw file has no artificial missing values added. Cleaning operations are
implemented for incoming dirty data and exercised by automated tests. Exact
duplicate records are removed; imputation and outlier limits are fitted only
on the training partition. See `metrics/data_quality.json` after a run.
