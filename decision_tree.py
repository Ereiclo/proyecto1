import numpy as np
from collections import Counter


class Nodo:
 # Defina cuales será sus mimbros datos
    def __init__(self, index):
        # Inicializar los mimbros datos

        self.index = index
        self.node_class = None
        self.left = None
        self.right = None
        self.representative_data = None
        self.probs = []

    def dataIsTerminal(Y):

        # retur true if this node have the sames labeles in Y
        count = Counter(Y)

        return len(count) == 1
        # clases = {Y[0]}

        # for i in range(1, len(Y)):
        #     if Y[i] not in clases:
        #         return False

        # return True

    def splitByValue(X, value):
        indexes_left = []
        indexes_right = []

        for i in range(len(X)):
            elem = X[i]
            if elem <= value:
                indexes_left.append(i)
            else:
                indexes_right.append(i)
        return indexes_left, indexes_right

    def BestSplit(X, Y):
        # write your code here
        _, dim = X.shape
        max_global = -np.Infinity
        selected_dim = -1
        split_for_dim = -1

        for j in range(dim):

            max_entropy_split = -np.Infinity
            selected_split = -1

            # print(X)
            # print(X[:,j][:10])
            # print({elem for elem in X[:,j][:10]})
            column_values = {elem for elem in X[:, j]}

            # print(f'Probando con dim {j}:')

            for elem in column_values:
                left, right = Nodo.splitByValue(X[:, j], elem)
                Y_left = Y[left]
                Y_right = Y[right]

                div_left = len(Y_left)/len(Y)  # i - 0 + 1
                div_right = len(Y_right)/len(Y)  # n - 1 - i

                # print(f'Division con {elem} da {len(Y_left)} a la izquierda y {len(Y_right)} a la derecha (desorden actual = {max_entropy_split})')
                # print(Y_left)
                # print(Y_right)
                # print(f'Entropia izquierda: {  Nodo.Entropy(Y_left)} {np.sum(Y_left)}')
                # print(f'Entropia derecha: {  Nodo.Entropy(Y_right)} {np.sum(Y_right)}')

                actual_disorder = Nodo.Entropy(
                    Y) - (div_left * Nodo.Entropy(Y_left) + div_right*Nodo.Entropy(Y_right))
                # print(f'Resultado para la entropia ({elem}): {actual_disorder}')

                if actual_disorder > max_entropy_split:
                    max_entropy_split = actual_disorder
                    selected_split = elem
                    # selected_split = -1 if div_left == 1 or div_right == 1 else elem

            # print(f'Elegido {selected_split} con {max_entropy_split} de ganacia')
            if max_entropy_split > max_global:
                selected_dim = j
                split_for_dim = selected_split
                max_global = max_entropy_split

        return selected_dim, split_for_dim


    def countByClass(Y):
        count = {}

        for class_ in Y:
            if class_ in count:
                count[class_] += 1
            else:
                count[class_] = 1
        return count

    def Entropy(Y):
        # write your code here
        if len(Y) == 0:
            return 0

        # count = Nodo.countByClass(Y)
        count = Counter(Y) 

        result = 0

        for c in count:
            pc = count[c]/len(Y)
            result = result + pc*np.log2(pc)

        return -result


class DT:
 # Defina cuales será sus mimbros datos

    def __init__(self, clases):
        # Inicializar los mimbros datos
        self.m_Root = None
        self.clases = clases

    def train(self, X, Y, *args):
        self.m_Root = Nodo(None)
        self.buildDTForCurrentNode(self.m_Root, X, Y)

    def buildDTForCurrentNode(self, actual_node, X, Y):
        # write your code here
        # print(Y)
        # print(X.shape)

        if not Nodo.dataIsTerminal(Y):
            # print(f'Best split para {len(Y)} elementos')
            selected_dim, split_for_dim = Nodo.BestSplit(X, Y)

            left, right = Nodo.splitByValue(X[:, selected_dim], split_for_dim)

            if len(left) == 0 or len(right) == 0:
                # print('caso donde se puede dividir el nodo aunque no sea terminal')
                # for elem in zip(X,Y):
                    # print(elem)
                # print(Nodo.BestSplit2(X,Y))
                # count = Nodo.countByClass(Y)
                # actual_node.node_class = max(count, key=count.get)

                count = Counter(Y)
                actual_node.node_class = count.most_common()[0][0]
                actual_node.probs = []
            
                for class_ in self.clases:
                    actual_node.probs.append(count[class_]/count.total())

                # print(actual_node.probs)


            else:

                actual_node.representative_data = split_for_dim
                actual_node.left = Nodo(None)
                actual_node.right = Nodo(None)
                actual_node.index = selected_dim
                self.buildDTForCurrentNode(
                    actual_node.left, X[left], Y[left])
                self.buildDTForCurrentNode(
                    actual_node.right, X[right], Y[right])

        else:
            actual_node.node_class = Y[0]
            actual_node.probs = [0 for _ in range(len(self.clases))]
            index_for_class = self.clases[Y[0]]
            actual_node.probs[index_for_class] = 1

    def predict(self, X):
        results = []

        for point in X:
            results.append(self.pointFindLeave(self.m_Root, point).node_class)

        return np.array(results)

    def pointFindLeave(self, actual_node, point):
        # print(actual_node.representative_data,actual_node.index,actual_node.node_class)
        if not (actual_node.node_class is None):
            return actual_node
        else:
            status = point[actual_node.index] <= actual_node.representative_data
            return self.pointFindLeave(actual_node.left, point) if status else self.pointFindLeave(actual_node.right, point)

    def probPredict(self,X):
        results = []

        for point in X:
            prob = self.pointFindLeave(self.m_Root,point).probs
            results.append(prob)
    
        return np.array(results)
    





# dt = DT(iris_np)






# dt = DT(X_train,Y_train)




# data = data[data[:,j].argsort()]
# testeo = data_test.to_numpy()


# testeo_sorted = testeo[testeo[:,1].argsort()]
# sorted_indexes = X_test[:,1].argsort()

# X_test_sorted = X_test[sorted_indexes]
# Y_test_sorted = Y_test[sorted_indexes]

# # print(data.head())
# # print(data.head().drop(columns='Clase'))
# print(X_test_sorted )
# print( testeo_sorted[:,:2])

# print(data.head()['Clase'])


# X0 = np.array(data['C1'])
# X1 = np.array(data['C2'])

# X = np.transpose(np.array([X0,X1]))
# Y = np.array(data['Clase'])
