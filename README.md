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

## Project 3 - Tech Stack Recommender (Content-Based Filtering)

A recommendation engine that maps a user's skills to the closest-matching job roles, using
TF-IDF vectorization and cosine similarity - no training, no model, just similarity math.

Dataset (`raw_skills.csv`, hand-built) - 10 job roles, each represented as a string of the
skills associated with it (e.g. Data Scientist: Python, SQL, Machine Learning, Statistics...).

**Pipeline:**
1. Fit `TfidfVectorizer` on the job roles' skill strings. This builds a shared vocabulary
   and down-weights common skills (like Python, which appears across many roles) while
   up-weighting rare, specific ones.
2. Take the user's input (comma-separated skills, minimum 3), clean and rejoin it into a
   single string in the same format as the job role rows.
3. Transform the user's string using the **same already-fitted vectorizer** - not a new
   one. This is the part most tutorials get wrong: TF-IDF needs a shared vocabulary to
   compare against, so the user's input has to be scored against the vocabulary learned
   from the job roles, not its own.
4. Compute cosine similarity between the user's vector and every job role's vector -
   this measures the angle between them, not their size, so it isn't skewed by how many
   skills someone lists.
5. Sort descending, return the top 3.

**Example run:**
```
enter your skills separated by commas: Python, Docker, Machine Learning

Based on your skills (Python Docker Machine Learning), here are your top matches:

Data Scientist - 0.505
ML Engineer - 0.485
Cloud Architect - 0.223
```

Data Scientist and ML Engineer both share Python + Machine Learning with the input, so
they correctly rank above Cloud Architect, which only overlaps on Docker.

**Note on the dataset:** the internship materials referenced a `raw_skills.csv` that
wasn't available in the shared resources, so this one was built by hand to match the
job roles and skills used in the project brief.

**Run:**
```
cd project-3-tech-stack-recommender
python tech_stack_recommender.py
```

Needs `pandas`, `scikit-learn`. Keep `raw_skills.csv` in the same folder as the script.


