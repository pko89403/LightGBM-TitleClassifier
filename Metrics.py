from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np 
import pprint
def top_n_result(pred, n):
    result = []
    for i in range(n):
        idx = np.argmax(pred)
        result.append(idx)
        pred = np.delete(pred, idx)
    return result 

def top_n_accuracy(y_true, y_pred, n, n_of_classes = 17):
    comp = []
    per_class = {}

    for c in range(n_of_classes):
        per_class[c] = []

    for act, pred  in zip(y_true, y_pred):
        top_n = top_n_result(pred, n)
        if act in top_n: 
            comp.append(1)
            per_class[act].append(1)
        else: 
            comp.append(0)
            per_class[act].append(0)
    
    return np.mean(comp), per_class

 #printing the predictions

def Report(y_true, y_pred, title, n_of_classes = 17):
    classes = [i for i in range(n_of_classes)]
    top_1 = [np.argmax(pred) for pred in y_pred]
    
    cfm = confusion_matrix(y_true=y_true, y_pred=top_1, labels=classes)
    acc = accuracy_score(y_true=y_true, y_pred=top_1)
    report = classification_report(y_true=y_true, y_pred=top_1, labels=classes)
    
    top1_acc, top1_acc_class = top_n_accuracy(y_true, y_pred, 1)
    top3_acc, top3_acc_class = top_n_accuracy(y_true, y_pred, 3) 
    top5_acc, top5_acc_class = top_n_accuracy(y_true, y_pred, 5)
    
    #print(cfm)
    #print(acc)
    with open(title + '.report', 'w') as f:
        f.write("----- Report -----\n")
        f.write(f"----- {title} -----\n")
        f.write(report)
        f.write('\n')
        f.write("------------------\n")
        f.write(f"Top 1's Accuarcy : Total - {top1_acc}")
        for c in range(n_of_classes):   f.write(f" {c} \t {np.mean(top1_acc_class[c])} \n")
        f.write(f"Top 3's Accuarcy : Total - {top3_acc}")
        for c in range(n_of_classes):   f.write(f" {c} \t {np.mean(top3_acc_class[c])} \n")    
        f.write(f"Top 5's Accuarcy : Total - {top5_acc}")
        for c in range(n_of_classes):   f.write(f" {c} \t {np.mean(top5_acc_class[c])} \n")    
        f.write("------------------")
