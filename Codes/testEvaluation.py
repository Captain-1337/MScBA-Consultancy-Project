# Precision
# P = correct emotion detected / correct emotion detected + detected but actually false

# Recall
# P = correct emotion detected / correct emotion detected + false detected but actually false

# F1
# 2 * ( (P * R) / (P + R) )

def precision(y_true, y_pred):
    i = set(y_true).intersection(y_pred)
    len1 = len(y_pred)
    if len1 == 0:
        return 0
    else:
        return len(i) / len1


def recall(y_true, y_pred):
    i = set(y_true).intersection(y_pred)
    return len(i) / len(y_true)


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0
    else:
        return 2 * (p * r) / (p + r)


if __name__ == '__main__':
    print(f1(['A', 'B', 'C'], [ 'A']))
    print(precision(['A', 'B', 'C'], [ 'A']))
    print(recall(['A', 'B', 'C'], [ 'A']))