# Dataset: UCI Spambase

- **Source:** UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/94/spambase
- **DOI:** https://doi.org/10.24432/C53G6X
- **Creators:** Mark Hopkins, Erik Reeber, George Forman, Jaap Suermondt (Hewlett-Packard Labs, 1999)
- **License:** CC BY 4.0

## File

`spambase.data` is a comma-separated file with **4,601 rows** (emails) and **58 columns**, with no header row.

| Columns | Count | Meaning |
|---|---|---|
| 1–48 | 48 | `word_freq_WORD`: percentage of words in the email that match WORD (e.g. free, money, remove, hp, george) |
| 49–54 | 6 | `char_freq_CHAR`: percentage of characters that match `; ( [ ! $ #` |
| 55–57 | 3 | Capital-letter runs: average length, longest run, total number of capital letters |
| 58 | 1 | Label: **1 = spam**, 0 = ham (not spam) |

Class balance: 1,813 spam (39.4%) and 2,788 ham (60.6%). The file contains 391 duplicate rows, which our code removes before training.
