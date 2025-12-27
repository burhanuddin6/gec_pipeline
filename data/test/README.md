## Filtering Problem Overview

We work with pairs of grammatical error correction (GEC) sentences.
The goal of the filtering algorithm is to **remove bad / noisy sentence pairs** while **retaining as many valid pairs as possible**.

Two types of errors are possible:

* **False negatives**: bad samples that are *not* filtered (noise leaks through)
* **False positives**: good samples that are *incorrectly filtered* (useful data is lost)

This makes the problem a **binary classification task**.

---

## Definitions

### Ground truth

* **Bad sample** → invalid or noisy GEC pair (should be filtered)
* **Good sample** → valid GEC pair (should be kept)

### Model decision

* **Filter**
* **Keep**

We treat **bad samples as the positive class**, since the algorithm’s primary job is to detect and remove them.

---

## Confusion Matrix

|                 | Filtered            | Kept                |
| --------------- | ------------------- | ------------------- |
| **Bad sample**  | True Positive (TP)  | False Negative (FN) |
| **Good sample** | False Positive (FP) | True Negative (TN)  |

---

## Metrics and Their Meaning

### Precision (Bad-sample precision)

> Of the samples that were filtered, how many were truly bad?

Precision = TP/(TP + FP)

* High precision → few **good samples are wrongly filtered**
* Measures how *trustworthy* the filtering decisions are

---

### Recall (Bad-sample recall)

> Of all bad samples, how many were successfully filtered?

Recall = TP/(TP + FN)

* High recall → few **bad samples slip through**
* Measures how *aggressive and effective* the filter is

---

### Accuracy (not very informative here)

Accuracy = (TP + TN)/(TP + TN + FP + FN)

* Can be misleading when good samples dominate the dataset
* A model that keeps everything can have high accuracy but zero recall on bad samples
* **Not a reliable metric for filtering quality**

---

## Practical Takeaway

* **Recall** controls how much noise remains in the data
* **Precision** controls how much clean data is lost
* There is an inherent tradeoff between the two

Most tuning decisions should focus on **precision–recall balance**, not accuracy.

