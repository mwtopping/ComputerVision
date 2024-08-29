import numpy as np
from astropy.table import Table







def gini(inds, targets):
    yes = [x for ii, x in enumerate(targets) if inds[ii]]
    return 1 -(np.sum(yes) / len(yes))**2 - ( (len(yes)-np.sum(yes)) / len(yes))**2


#    return 1-(len(leftinds)/(len(leftinds)+len(rightinds)))**2 - (len(rightinds)/(len(leftinds)+len(rightinds)))**2



def min_gini(arr, targets):
    sortedarr = np.array(sorted(arr))

    bounds = 0.5*(sortedarr[:-1]+sortedarr[1:])

    ginis = []

    for ii, bound in enumerate(bounds):
        linds = arr <= bound 
        rinds = arr > bound 
        ginil = gini(linds, targets)*sum(linds)/len(arr)
        ginir = gini(rinds, targets)*sum(rinds)/len(arr)

        ginis.append(ginil+ginir)

    return bounds[np.argmin(ginis)], np.min(ginis)




def build_tree(data, target, labels):

    tree = {'label':None, 'sepval':None, 'left':None, 'right':None}
    bestlab = None
    mingini = np.inf
    sepval = 0
    for lab in labels:
        l, g = min_gini(data[lab], data[target])
        if g < mingini:
            mingini = g
            bestlab = lab
            sepval = l
    remaininglabs = [x for x in labels if x is not bestlab]

    tree['label'] = bestlab
    tree['sepval'] = sepval

    leftdata = data[data[bestlab] <= sepval]
    rightdata = data[data[bestlab] > sepval]

    if len(remaininglabs) > 0:
        tree['left'] = build_tree(leftdata, target, remaininglabs)
        tree['right'] = build_tree(rightdata, target, remaininglabs)
    else:
        print("done with remaining label ", remaininglabs)
        tree['left'] =  {'label':"Leaf", 'value':np.sum(data[target])/len(data[target])}
        tree['right'] =  {'label':"Leaf", 'value':(len(data[target]) - np.sum(data[target]))/len(data[target])}
    
    return tree



def traverse(tree, values):
    if tree['label'] == "Leaf":
        print(tree)
    else:
        thislabel = tree['label']
        print(tree['label'], tree['sepval'], values[thislabel])
        if values[thislabel] <= tree['sepval']: 
            print("going left")
            traverse(tree['left'], values)
        else:
            print("going right")
            traverse(tree['right'], values)



if __name__ == "__main__":
    data = Table.read('./heart.csv', format='csv')
    print(data)
    target = 'target'
    labels = ['age', 'trestbps', 'chol']

    tree = build_tree(data, target, labels)

    test = {'age':55, 'trestbps':145, 'chol':250}
    traverse(tree, test)
