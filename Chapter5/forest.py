from decisiontree import *
import matplotlib.pyplot as plt
from tqdm import tqdm






if __name__ == "__main__":
    data = Table.read('./heart.csv', format='csv')
    target = 'target'
    labels = np.array(['age', 'trestbps', 'chol', 'oldpeak', 'thalach'])

    accuracy = []

    for kk in range(50):
        Ndata = int(len(data)*0.8)
        training = np.random.choice(range(len(data)), size=Ndata, replace=False)
        testing = [x for x in range(len(data)) if x not in training]

        trainingdata = data[training]
        testingdata = data[testing]


        forest = []
        Ntrees = 100
        for ii in tqdm(range(Ntrees)):
            selecteddata = np.random.choice(range(Ndata), size=Ndata, replace=True)
            selectedlabels = np.random.choice(range(len(labels)), size=3, replace=False)

            _trainingdata = trainingdata[selecteddata]
            tree = build_tree(_trainingdata, target, labels[selectedlabels])

            forest.append(tree)


        Ncorr =0
        Nerr = 0

        for obj in testingdata:
            res = 0
            for t in forest:
                val = traverse(t, obj)
        
                res += val
            finalresult = round(res/Ntrees)
            if finalresult == obj[target]:
                Ncorr += 1
            else:
                Nerr += 1

        print(f"Number correct: {Ncorr}")
        print(f"Number incorrect: {Nerr}")
        accuracy.append(Ncorr/(Nerr+Ncorr))

fig, ax = plt.subplots()
ax.hist(accuracy, bins=np.linspace(0, 1, 30))
plt.show()

