
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from Describe_2 import Describe2
import os
os.chdir("C:/Users/AE250016/Desktop/ACA_DS/Untitled Folder")


class LoadingEda:

    @staticmethod
    def prepare_data(df, y_label, test_split, k_folds):
        """
         Args:
        df: input pandas data frame with all the data
        y_label: string with the target column name
        test_split: percentage of test split
        k_folds: if integer then states the number of cross fold from training data to evaluate the model
        if float then states the percentage of training data to be used for validation

        Returns: if k_folds integer :
        x_train : numpy array for training data
        x_test  : numpy array for testing data
        y_train : numpy array for training target labels
        y_test  : numpy array for testing target labels
        x_array : numpy array for all the data
        y_array : numpy array for all target labels
        """
        df_dummies = pd.get_dummies(df.loc[:, df.columns != y_label].fillna(0))
        x_array = df_dummies.values
        y_array = df[[y_label]].values.ravel()
        if isinstance(k_folds, float):
            x_train_f, x_test, y_train_f, y_test = train_test_split(x_array, y_array, test_size=0.2, random_state=145)
            x_train, x_val, y_train, y_val = train_test_split(x_train_f, y_train_f, test_size=0.2, random_state=145)
            return x_train, x_test, y_train, y_test, x_array, y_array, x_val, y_val
        else:
            x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, random_state=145,
                                                                test_size=test_split)
            return x_train, x_test, y_train, y_test, x_array, y_array

    def __init__(self, df, y_label, k_folds, test_split=0.2):
        """

        :param df: input pandas data frame with all the data
        :param y_label: string with the target column name
        :param test_split: percentage of test split
        :param k_folds: if integer then states the number of cross fold from training data to evaluate the model
        if float then states the percentage of training data to be used for validation
        """
        if isinstance(k_folds, float):
            self.x_train, self.x_test, self.y_train, self.y_test, self.x_array, self.y_array, self.x_val, self.y_val = self.prepare_data(df, y_label, k_folds, test_split)
        else:
            self.x_train, self.x_test, self.y_train, self.y_test, self.x_array, self.y_array = self.prepare_data(df, y_label, k_folds, test_split)
        self.numeric, self.categorical = Describe2.d2(Describe2(), df)
        self.df = df

    #     def box_plots(self):
    #         melted_df = pd.melt(self.df[list(self.numeric[self.numeric['No. of NaN'] == 0].index)])
    #         h = sns.FacetGrid(col="variable", row="value", data= melted_df)
    #         h.map(plt.hist, "variable")

    def plot_dist(self):  # doesn't plot any columns with none values
        """
    This is a self envoked function that creates a pairplot for all non-null numeric columns
        """
        sns.pairplot(self.df[list(self.numeric[self.numeric['No. of NaN'] == 0].index)])

    def corr_heat_map(self):
        """
    This is a self envoked function that creates a correlation heat map for all non-null numeric columns    
        """
        corr = self.df[list(self.numeric[self.numeric['No. of NaN'] == 0].index)].corr()
        sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)


titanic = pd.read_csv('train.csv')
titanic = titanic[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

loading_EDA_obj = LoadingEda(titanic, 'Survived', 0.2, 0.2)
