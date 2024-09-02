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
    return bounds[np.nanargmin(ginis)], np.nanmin(ginis)




def build_tree(data, target, labels):

    tree = {'label':None, 'sepval':None, 'left':None, 'right':None}
    bestlab = None
    mingini = np.inf
    sepval = 0

    if len(data) == 1:
        return {'label':"Leaf", 'value':data[0][target]}

    for lab in labels:

        if len(list(set(data[lab]))) <= 1:
            value = round(np.sum(data[target])/len(data))
            return {'label':"Leaf", 'value':value}


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
        tree['left'] =  {'label':"Leaf", 'value':round(np.sum(data[target])/len(data[target]))}
        tree['right'] =  {'label':"Leaf", 'value':round((len(data[target]) - np.sum(data[target]))/len(data[target]))}
    
    return tree



def traverse(tree, values):
    if tree['label'] == "Leaf":
        return tree['value']
    else:
        thislabel = tree['label']
        if values[thislabel] <= tree['sepval']: 
            val = traverse(tree['left'], values)
        else:
            val = traverse(tree['right'], values)

    return val



if __name__ == "__main__":
    data = Table.read('./heart.csv', format='csv')
    print(data)
    target = 'target'
    labels = ['age', 'trestbps', 'chol', 'thalach']

    tree = build_tree(data, target, labels)
    print(tree)
    print(traverse(tree, data[0]))
