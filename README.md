# DecodeLabs Internship 

Projects from the DecodeLabs AI internship 


 Project 1 - Rule-Based Chatbot

A command line chatbot that matches user input against a dictionary of predefined
responses. No learning involved the logic is written by hand, which is the point

Input is lowercased and stripped before matching so "Hello", "hello " and "HELLO"
all hit the same key. Unrecognised input falls back to a default reply via
`dict.get()`. Typing `exit` breaks the loop.
**Run:**
```
cd project-1-chatbot
python chatbot.py
```


**Limitation worth noting:** it only matches exact strings. "how r u" fails even
though "how are you" works. Fixing that properly needs intent matching rather than
a lookup table, which is where actual ML starts.



## Project 2 - Data Classification (Iris + KNN)

Supervised classification on the Iris dataset - 150 samples, 4 features, 3 balanced
classes. K-Nearest Neighbors as the main model.

Pipeline: load → stratified 80/20 split → StandardScaler → KNN (k=5) → evaluate.

**Results:**

| Metric | Value |
|---|---|
| Accuracy | 96.67% |
| F1 (macro) | 0.9666 |

Confusion matrix:
```
[[10  0  0]
 [ 0  9  1]
 [ 0  0 10]]
```


One error out of 30. Setosa is perfectly separable and gets 10/10. The single
mistake is a versicolor predicted as virginica those two overlap in petal
measurements. Same error read two ways: versicolor recall drops to 0.90, virginica
precision drops to 0.91.

**Extra experiments:**

- Choosing K - used 5 fold cross validation across k=1 to 30 instead of
  defaulting to 5. Best K came out as 5 anyway, at 0.975 CV accuracy.
- Does scaling matter here? trained KNN with and without scaling. Identical
  result, 0.9667 both ways. All four features are already in similar cm ranges, so
  standardizing doesn't reorder any neighbours. Scaling is still mandatory for
  distance based models in principle, but its measurable effect depends on whether
  the feature ranges actually differ.
- Algorithm comparison - KNN, Logistic Regression, Decision Tree and SVM all
  tie at 0.9667, all making the same single error. When four different algorithms
  plateau at the same number, the limit is the data not the model.

**Caveat:** 30 test samples means one error is worth 3.3 percentage points, so these
comparisons sit inside the noise. That's why K was picked with cross validation
rather than a single split.

**Run:**
```
cd project-2-iris-classification
python iris_classification.py
```

Needs `scikit-learn`, `numpy`, `matplotlib`.



