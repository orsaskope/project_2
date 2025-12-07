# Εργασία 2 — Neural LSH  
**Βασιλική-Όρσα Σκοπελίτη — 1115201900341**  
**Γρηγόρης Ταμπαξής — 1115202000188**  
**Μάθημα: Ανάπτυξη Λογισμικού για Αλγοριθμικά Προβλήματα**

---

## Ενδεικτικός τρόπος εκτέλεσης `nlsh_build.py`

```bash
python3 nlsh_build.py -d input.dat -i nlsh_index -type mnist --knn 15 -m 100 --epochs 10 --nodes 256
```

## Ενδεικτικός τρόπος εκτέλεσης `nlsh_search.py`

```bash
python3 nlsh_search.py -d input.dat -q query.dat -i nlsh_index -o output.txt -type mnist -N 10 -T 10
```

---

# Περιγραφή

Στην εργασία αυτή υλοποιείται ο αλγόριθμος **Neural LSH**, ο οποίος μαθαίνει μια βελτιστοποιημένη διαμέριση του χώρου των δεδομένων, συνδυάζοντας:

- Κατασκευή γράφου k-Nearest Neighbors (k-NN)
- Μετατροπή σε weighted undirected graph
- Ισοκατανεμημένη διαμέριση μέσω KaHIP
- Εκπαίδευση MLP classifier (PyTorch) για πρόβλεψη block
- Δημιουργία inverted file για γρήγορη αναζήτηση

Η διαδικασία χωρίζεται σε:

- **nlsh_build.py** (build phase)  
- **nlsh_search.py** (search phase)

Επιπλέον, έχουν γίνει βελτιώσεις στον αλγόριθμο **IVFPQ** της πρώτης εργασίας για σωστή συγκριτική αξιολόγηση.

Τα αρχεία python αυτης της εργασίας ειναι:
---

* **nlsh_build.py**(υπευθυνο για το χτισιμο του nlsh με την χρηση του dataset)
* **nlsh_search.py**(υπευθυνο για την αναζητηση τον Query vectors)
* **graph_utils.py**(υπευθυνο για την δημιουργια του knn γραφου,του weighted undirected γραφου και του csr , ειτε σε brute force ειτε σε ivfflat)
* **models.py**(υπευθυνο για την δημιουργια και την εκπαιδευση του μοντελου)
* **data_parser.py**(υπευθυνο για την δημιουργια των σωστων μεταβλητων για την ασκηση,υλοποιηση ειναι ηδια σε λογικη με την 1η ασκηση απλα υλοποιειται σε python αντι c++)

Σημαντικες υποσημειωσεις
---

* Παρότι μπορουν να χρησιμοποιηθουν δυο τροποι (Brute force, IVFFLAT) για δημιουργια build, προτείνουμε

  * Χρηση brute force αν επιθυμειτε να χρησιμοποιησετε subset για γρηγορα αποτελεσματα (καθως με την χρηση της μεταβλητης debug_x μπορειτε να ελεγξετε ποσα vectors απο το dataset θα χρησιμοποιηθουν).
  * Χρηση ivfflat αν επιθυμειτε να χρησιμοποιησετε ολο το dataset
* Αμα επιλεχθεί brute force οι μεταβλητες debug_x πρεπει να εχουν ιδια τιμη στο nlsh_build και nlsh_search για να λειτουργησει σωστα το προγραμμα.



---

# 1. Κατασκευή Γράφου k-NN

Η συνάρτηση `get_choice()` επιτρέπει επιλογή ανάμεσα σε:

1. Brute force  
2. IVFFLAT (C++ κώδικας από 1η εργασία)

και επιστρέφει τη μέθοδο κατασκευής.

---

# 2. nlsh_build

## load_dataset(p, choice)
- Διαβάζει MNIST/SIFT
- Κάνει κανονικοποίηση (/255 ή /218)
- Επιστρέφει `X` και `N`

---

## build_knn_bruteforce(X, p)
Καλεί διαδοχικά:

1. `compute_knn`
2. `build_weighted_knn_graph`
3. `knn_graph_to_csr`

και επιστρέφει **xadj, adjncy, adjcwgt, vwgt**.

---

## build_knn_ivfflat(p, N)
- Τρέχει IVFFLAT μέσω subprocess  
- IVFFLAT γράφει ANN αποτελέσματα στο `tmp.txt`
- Έπειτα καλεί:
  - `bwg_ivfflat`
  - `build_csr_ivfflat`
- Επιστρέφει επίσης **xadj, adjncy, adjcwgt, vwgt**

---

## run_kahip(p, vwgt, xadj, adjcwgt, adjncy)
Καλεί τη συνάρτηση:

```
kahip.kaffpa(...)
```

και επιστρέφει:

- **edgecut**  
- **blocks** (το partition assignment για κάθε σημείο)

---

## train_mlp(X, blocks, p)

- Δημιουργεί PyTorch Dataset + DataLoader
- Ορίζει MLP (layers, nodes, dropout, batchnorm)
- Εκπαιδεύει μοντέλο με epochs & learning rate
- Το MLP μαθαίνει τη διαμέριση του KaHIP

---

## build_inverted_lists(blocks, m)
Φτιάχνει inverted file:

```
for κάθε block x -> λίστα με indices των vectors που ανήκουν σε αυτό
```

---

## save_index(...)
Αποθηκεύει:

- trained MLP
- blocks (labels)
- inverted lists
- metadata (m, dim, params)

Το index χρησιμοποιείται στο search.

---


## main()
Εκτελεί όλο το build pipeline:

1. load_dataset  
2. build_knn  
3. run_kahip  
4. train_mlp  
5. build_inverted_lists  
6. save_index  

---

# 3. graph_utils

Περιλαμβάνει:

- κατασκευή γράφου  
- υπολογισμό βαρών  
- μετατροπή σε CSR arrays  

---

## Κατηγορία 1 — Brute Force

### compute_knn
Υπολογίζει k-NN για κάθε σημείο (directed graph).

### build_weighted_knn_graph

Λογική βαρών:

| Περίπτωση | Weight |
|----------|--------|
| j ∈ knn(i) **και** i ∈ knn(j) | 2 |
| αλλιώς αν μονο ενα ειναι γείτονας του άλλου | 1 |

### knn_graph_to_csr
Φτιάχνει:

- xadj  
- adjncy  
- adjwgt  
- vwgt  

---

## Κατηγορία 2 — IVFFLAT

### bwg_ivfflat(path, N)
- Διαβάζει το `tmp.txt`
- Φτιάχνει weighted undirected graph
- Αν κόμβος δεν εμφανιστεί → δημιουργείται άδεια λίστα

### build_csr_ivfflat
Φτιάχνει CSR arrays για KaHIP.

---

# 4. nlsh_search

## load_index
Φορτώνει:

- MLP  
- blocks  
- inverted lists  
- metadata  

---

## brute_force_search
Υπολογίζει τον true NN (για evaluation).

---

## top_t_bins(model, query, T)
- Μετατρέπει query σε tensor
- Περνάει μέσα από MLP
- Softmax → πιθανότητες
- Επιστρέφει τα **T καλύτερα bins**

---

## search_in_bins
Πραγματοποιεί ANN αναζήτηση **μόνο** στα T bins.

---

## nlsh_search

Περιλαμβάνει:

1. Φόρτωση dataset & queries  
2. Κανονικοποίηση  
3. Φόρτωση index  
4. `model.eval()`  
5. Για κάθε query:
   - brute force NN  
   - top T bins  
   - ANN search σε αυτά  
   - range search (αν ζητηθεί)

Το output είναι συμβατό με την εργασία 1.

# 5. Models

## MLPClassifier
* Απλό Multi-Layer Perceptron (MLP) με δυνατότητα Dropout και BatchNorm.  
Χρησιμοποιείται για να προβλέπει σε ποιο bin ανήκει κάθε vector.

## get_device()
* Επιστρέφει το κατάλληλο device για PyTorch (GPU αν υπάρχει, αλλιώς CPU).

## build_dataloader(X, y, batch_size=32)
* Μετατρέπει τα δεδομένα σε PyTorch Dataset και επιστρέφει DataLoader για batching και shuffle.

## train_model(model, loader, epochs=10, lr=1e-3)
* Εκπαιδεύει το MLP χρησιμοποιώντας Adam optimizer και CrossEntropyLoss.
* Επιστρέφει το εκπαιδευμένο μοντέλο.

# 6. Dataparser

* Με ακριβώς ιδια λογική με την ασκηση 1 δημιουργει τις σωστες μεταβλητες με default τιμες για την υλοποιηση της εργασιας ενω επισης ειναι υπευθυνο για την σωστη αναγνωση των δεδομενων sift και mnist.

* Η μονη διαφορα απο την 1η εργασια ειναι οτι η υλοποιηση του τωρα εχει γινει σε python.


