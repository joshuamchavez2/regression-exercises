# Math 
from math import sqrt
from scipy import stats
import statistics
import os

# General
import numpy as np
import pandas as pd

# Sklearn Modeling
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.preprocessing

# Visuals
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Graph
from tabulate import tabulate

# Custom Module's
from wrangle import wrangle_zillow

import warnings
warnings.filterwarnings("ignore")





############################# Explore #############################


def plot_variable_pairs(train, cols, descriptive=None, hue=None):
    '''
    This function takes in a df, a list of cols to plot, and default hue=None 
    and displays a pairplot with a red regression line. If passed a descriptive
    dictionary, converts axis titles to the corresponding names.
    '''
    # sets line-plot options and scatter-plot options
    keyword_arguments={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}}
    
    # creates pairplot object
    pairplot = sns.pairplot(train[cols], hue=hue, kind="reg",\
            plot_kws=keyword_arguments)
    
    # if passed a descriptive dictionary, iterates through matplotlib axes
    # in our pairplot object and sets their axis labels to the corresponding 
    # strings.
    if descriptive:
        for ax in pairplot.axes.flat:
            ax.set_xlabel(descriptive[ax.get_xlabel()])
            ax.set_ylabel(descriptive[ax.get_ylabel()])
    
    # Adds a super-title
    pairplot.fig.suptitle('Correlation of Continuous Variables', y=1.08)
    plt.show()

def create_heatmap(train, cols, descriptive=None):
    corr_matrix = train[cols].corr()

    kwargs = {'alpha':.9,'linewidth':3, 'linestyle':'-', 
          'linecolor':'k','rasterized':False, 'edgecolor':'w', 
          'capstyle':'projecting',}
    labels = pd.Series(cols)
    if descriptive:
        labels = labels.map(descriptive)
    plt.figure(figsize=(8,6))
    heatmap = sns.heatmap(corr_matrix, cmap='Purples', annot=True, \
                          xticklabels=labels, yticklabels=labels, vmin=0, vmax=1, **kwargs)
    plt.ylim(0, 3)
    plt.title('Correlation of Continuous Variables')
    plt.show()

def plot_categorical_and_continuous_vars(categorical_vars, continuous_vars, df, descriptive=None):
    """
    This function that takes in a string name of a categorical variable, 
    a string name from a continuous variable and the df they live in and
    displays 4 different plots.
    """
    for categorical_var in categorical_vars:
        for continuous_var in continuous_vars:
            categorical_label = categorical_var
            continuous_label = continuous_var
            if descriptive:
                categorical_label = descriptive[categorical_var]
                continuous_label = descriptive[continuous_var]
            
            fig, axes = plt.subplots(figsize=(12,36), nrows=4,ncols=1)
            fig.suptitle(f'{continuous_label} by {categorical_label}', fontsize=18, y=1.02)
            sns.lineplot(ax=axes[0], x=categorical_var, y=continuous_var, data=df)
            axes[0].set_title('Line Plot', fontsize=14)
            axes[0].set_xlabel(categorical_label, fontsize=12)
            axes[0].set_ylabel(continuous_label, fontsize=12)
            
            sns.boxplot(ax=axes[1], x=categorical_var, y=continuous_var, data=df,\
                        color='blue')
            axes[1].set_title('Box-and-Whiskers Plot', fontsize=14)
            axes[1].set_xlabel(categorical_label, fontsize=12)
            axes[1].set_ylabel(continuous_label, fontsize=12)
            
            sns.swarmplot(ax=axes[2], x=categorical_var, y=continuous_var, data=df,\
                        palette='Blues')
            axes[2].set_title('Swarm Plot', fontsize=14)
            axes[2].set_xlabel(categorical_label, fontsize=12)
            axes[2].set_ylabel(continuous_label, fontsize=12)
            
            sns.barplot(ax=axes[3], x=categorical_var, y=continuous_var, data=df,\
                        palette='Purples')
            axes[3].set_title('Bar Plot', fontsize=14)
            axes[3].set_xlabel(categorical_label, fontsize=12)
            axes[3].set_ylabel(continuous_label, fontsize=12)
            
            plt.tight_layout()

            plt.show()

