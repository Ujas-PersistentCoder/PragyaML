"""Assumes pre processed data with no missing values"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Optional
from pragya.utils.metrics import gain

"""=============================================================================="""

@dataclass
class Node:

    feature: Optional[int] = None
    threshold: Optional[Any] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    value: Optional[Any] = None

    def is_leaf_node(self):
        """To check if a node is a leaf node or not"""
        return self.value is not None
    
    
class DecisionTree:

    def __init__(self, metric, max_depth = None, min_samples_for_split = 2):
        self.root = None
        self.metric = metric
        self.max_depth = max_depth
        self.min_samples_for_split = min_samples_for_split


    def _grow_tree(self, X, y, depth = 0):

        """Recursively grows the decision tree"""

        num_samples, num_features = X.shape
        unique_labels = np.unique(y)

        if ((len(unique_labels) == 1) or (num_samples < self.min_samples_for_split) or (self.max_depth is not None and depth >= self.max_depth)): 
            return Node(value = self._calculate_leaf_value(y))
        
        best_feature, best_threshold = self._best_split(X, y, num_features)

        if best_feature is None:
            return Node(value = self._calculate_leaf_value(y))
        
        feature_column = X[:, best_feature]
        if isinstance(best_threshold, (int, float, np.number)): #checks whether the feature is numerical
            left_indices = np.where(feature_column <= best_threshold)[0]
            right_indices = np.where(feature_column > best_threshold)[0]
        else:
            left_indices = np.where(feature_column == best_threshold)[0]
            right_indices = np.where(feature_column != best_threshold)[0]

        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature = best_feature, threshold = best_threshold, left = left_subtree, right = right_subtree)

    
    def _calculate_leaf_value(self, y):

        """Calculate the value for a leaf node"""

        if self.metric in ["entropy", "gini"]:
            values, counts = np.unique(y, return_counts=True)
            mode = values[np.argmax(counts)]
            return mode
        elif self.metric == "mse":
            mean_value = np.mean(y)
            return mean_value
        else:
            raise ValueError("Invalid metric. Valid metrics: entropy, gini, or mse")
        
    
    def _best_split(self, X, y, num_features):

        """Finds the best feature and threshold to split the data on based on the gain calculated using the specified metric"""

        best_gain = -1
        best_feature, best_threshold = None, None
        for feature in range(num_features):

            column  =  X[:, feature]
            unique_values = np.unique(column)
            is_numerical = isinstance(unique_values[0], (int, float, np.number))
            cardinality_ratio = len(unique_values) / len(column)
            
            if is_numerical and cardinality_ratio > 0.05:
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2 #midpoints, np.unique return sorted unique values
            else:
                thresholds = unique_values
            
            for threshold in thresholds:
                if is_numerical:
                    left_indices = np.where(column <= threshold)[0]
                    right_indices = np.where(column > threshold)[0]
                else:
                    left_indices = np.where(column == threshold)[0]
                    right_indices = np.where(column != threshold)[0]
                if len(left_indices) == 0 or len(right_indices) == 0: continue #skip invalid splits
                current_gain = gain(y, y[left_indices], y[right_indices], self.metric)
                if current_gain > best_gain:
                    best_gain = current_gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold
    
    def fit(self, X, y):

        """Fits the decision tree to the training data"""

        X = np.array(X)
        y = np.array(y)
        self.root = self._grow_tree(X, y)


    def _traverse_tree(self, x, node : Node):

        """Traverses the tree to make a prediction for a single data point"""

        if node.is_leaf_node(): return node.value
        if isinstance(x[node.threshold], (int, float, np.number)):
            if x[node.feature] <= node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        else:
            if x[node.feature] == node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
            

    def predict(self, X):

        """Predicts the labels/values for the entire input data"""

        X = np.array(X)
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)
    

"""=============================================================================="""
"""post pruning and handling missing values to be implemented later"""