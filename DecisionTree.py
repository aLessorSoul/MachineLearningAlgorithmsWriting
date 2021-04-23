import pandas as pd
import numpy as np
from collections import Counter

class Decision_Tree_Node:
    def __init__(self):
        self.children = []
        self.variable = None
        self.pred_class = None
        self.split_value = None
        self.is_categorical = False
        self.is_leaf = False
        
class Decision_Tree:
    def __init__(self):
        self.root = Decision_Tree_Node()

    def leaf_prediction(self, tree_node, y):
        tree_node.pred_class = max(Counter(y))
        tree_node.is_leaf = True
        
    def information_gain(self, mask, y): 
        def ent(y): #计算根据v分割后的y的信息熵
            count_dict = Counter(y)
            total = len(y)
            return np.sum([-p * np.log2(p) for p in [v/total for v in count_dict.values() if v != 0]])
        ent_after = np.sum([ent*l for ent, l in y.groupby(mask).agg(lambda y: (ent(y),len(y)))])/len(y)
        ent_before = ent(y)
        return ent_before - ent_after

    def gain_ratio(self, mask, y):
        def ent(y): #计算根据v分割后的y的信息熵
            count_dict = Counter(y)
            total = len(y)
            return np.sum([-p * np.log2(p) for p in [v/total for v in count_dict.values() if v != 0]])
        ent_after = np.sum([ent*l for ent, l in y.groupby(mask).agg(lambda y: (ent(y),len(y)))])/len(y)
        ent_before = ent(y)
        return (ent_before - ent_after)/ent_before
    
    def gini_gain(self, mask, y): 
        def gini(y): #计算根据v分割后的y的信息熵
            count_dict = Counter(y)
            total = len(y)
            return 1 - np.sum([p ** 2 for p in [v/total for v in count_dict.values() if v != 0]])
        gini_after = np.sum([gini*l for gini, l in y.groupby(mask).agg(lambda y: (gini(y),len(y)))])/len(y)
        gini_before = gini(y)
        return gini_before - gini_after
    
    def find_split_var(self, X, y, eval_fun, dtype_dict):
        evaluate_results = {}
        best_v = None
        best_value = None
        best_criteria = 0
        is_categorical = False
        for v, dtype in dtype_dict.items():
            evaluate_results[v] = {}
            if dtype == 'categorical':
                iterator = X[v].unique()[:-1]
                if len(iterator) < 1:
                    continue
                for value in iterator:
                    
                    mask = X[v] == value
                    evaluate_results[v][value] = eval_fun(mask, y)
                    if evaluate_results[v][value] > best_criteria:
                        best_v = v
                        best_value = value
                        best_criteria = evaluate_results[v][value]
                        is_categorical = True
                        
            elif dtype == 'numerical':
                num_values = X[v].unique()
                if len(num_values) < 2:
                    continue
                last_value = num_values[0]
                for current_value in num_values[1:]:
                    value = (last_value + current_value)/2
                    mask = X[v] >= value
                    evaluate_results[v][value] = eval_fun(mask, y)
                    if evaluate_results[v][value] > best_criteria:
                        best_v = v
                        best_value = value
                        best_criteria = evaluate_results[v][value]
            else:
                print('dtype error')
                raise
        #print(evaluate_results)
        return (best_v, best_value, best_criteria, is_categorical)
    
    def node_fit(self, tree_node, X, y, dtype_dict, criteria, prune):
        print('checking node', tree_node)
        if criteria not in ['information_gain', 'gain_ratio', 'gini_gain']:
            print('criteria invalid')
            raise
        else:
            eval_fun = {'information_gain':self.information_gain, 'gain_ratio':self.gain_ratio, 'gini_gain': self.gini_gain}[criteria]
        if prune == 'pre_prune':
            pass
            #BFS search
        else: #DFS search
            if dtype_dict == {}:
                self.leaf_prediction(tree_node, y)
                return None
            best_v, best_value, best_criteria, is_categorical = self.find_split_var(X, y, eval_fun, dtype_dict)

            if best_v is None:
                self.leaf_prediction(tree_node, y)
                return None
            else:
                tree_node.variable = best_v
                tree_node.split_value = best_value
                tree_node.is_categorical = is_categorical
                
            #left Child 小于
            if is_categorical:
                mask = X[best_v] != best_value
            else:
                mask = X[best_v] != best_value
                
            left_child = Decision_Tree_Node()
            self.node_fit(left_child, X.loc[mask,:], y[mask], dtype_dict, criteria = criteria, prune = prune)
            tree_node.children.append(left_child)
            
            #right child 大于等于
            right_child = Decision_Tree_Node()
            self.node_fit(right_child, X.loc[-mask,:], y[-mask], dtype_dict, criteria = criteria, prune = prune)
            tree_node.children.append(right_child)
            
    def fit(self, X, y, criteria = 'gini_gain', prune = None, dtype_dict = None):
        def check_data_type(X):
            dtype_dict={}
            for v in X.columns:
                if X[v].dtype in ['int','int64','float','float64']:
                    if len(X[v].unique()) <= 5:
                        dtype_dict[v] = 'categorical'
                    else:
                        dtype_dict[v] = 'numerical'
                else:
                    dtype_dict[v] = 'categorical'
            return dtype_dict
        if dtype_dict is None:
            dtype_dict = check_data_type(X)
        self.node_fit(self.root, X, y, dtype_dict, criteria, prune)
        
    def predict_one_sample(self, x):
        return self.predict_iterator(self.root, x)
        
    
    def predict_iterator(self, tree_node, x):
        if tree_node.us_leaf:
            return tree_node.predict_class
        elif tree_node.is_categorical:
            if x[tree_node.variable] == tree_node.split_value:
                return self.predict_iterator(tree_node.children[1], x)
            else:
                return self.predict_iterator(tree_node.children[0], x)
        else:
            if x[tree_node.variable] >= tree_node.split_value:
                return self.predict_iterator(tree_node.children[1], x)
            else:
                return self.predict_iterator(tree_node.children[0], x)
        
    def predict(self, X):
        return X.apply(self.predict_one_sample, axis = 1)
    
    
    def node_print(self, tree_node, indent): #打印node，支持dfs
        if tree_node.is_leaf:
            return
        elif tree_node.is_categorical:
            print(indent * '--', tree_node.variable,'is', tree_node.split_value, ':', tree_node.children[1].pred_class)
            self.node_print(tree_node.children[1], indent + 1)
            print(indent * '--', tree_node.variable,'is not', tree_node.split_value, ':', tree_node.children[0].pred_class)
            self.node_print(tree_node.children[0], indent + 1)
        else:
            print(indent * '--', tree_node.variable,'>=', tree_node.split_value, ':', tree_node.children[1].pred_class)
            self.node_print(tree_node.children[1], indent + 1)
            print(indent * '--', tree_node.variable,'<', tree_node.split_value, ':', tree_node.children[0].pred_class)
            self.node_print(tree_node.children[0], indent + 1)
            
    def print(self, indent = 0):  #invoke dfs打印node
        self.node_print(self.root, indent)

df = pd.read_csv('data.csv')
X = df.iloc[:,1:-2]
y = df['stroke']

d = Decision_Tree()
d.fit(X, y)
d.print()