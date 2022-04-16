'''
Collection of function to do Customer Churn analysis and modeling

Author: Emmanuel AKinwale
Date  : April 2022
'''

# import libraries
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from lists_vars import keep_columns
sns.set()
# from sklearn.preprocessing import normalize
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


class Visual:
    '''
    This is a class to visualize the data by performing EDA,
    classification report, and creating correlation matrix

    Parameters
    ----------
    df: pandas dataframe
        pandas dataframe to be visualized
    y_train: pandas series
        y_train to be used for classification report
    y_test: pandas series
        y_test to be used for classification report
    y_train_preds_rf: pandas series
        y_train_preds for random forest
    y_train_preds_lr: pandas series
        y_train_preds for logistic regression
    y_test_preds_lr: pandas series
        y_test
    '''

    def __init__(self):
        """
        Initialize the instance.
        """
    # plot all columns in the Quant_cols anmd Categorical_Cols list from the
    # dataframe (df)
    def perform_eda(self, df):
        """
        This function performs exploratory data analysis on the given dataframe.
        Parameters
        ----------
        df: a pandas dataframe object
        Returns
        -------
        None.
        """
        quant_columns = df.select_dtypes(include="number")
        cat_columns = df.select_dtypes(exclude="number")
        fig = plt.figure(figsize=(10, 10))
        plt.gca()
        plt.rcParams.update({'figure.max_open_warning': 0})
        for lst in quant_columns:
            df[str(lst)].hist()
            plt.title(lst)
            plt.savefig(
                "./images/eda/quantitative_plots/hist_%s.png"%lst)
            plt.close(fig)
        for lst in cat_columns:
            df[str(lst)].value_counts('normalize').plot(kind='bar')
            plt.title(lst)
            plt.savefig(
                "./images/eda/categorical_plots/bar_%s.png"%lst)
            plt.close(fig)
        # to plot the distribution and heatmap
        sns.distplot(df['Total_Trans_Ct'], kde=True)
        plt.title('Total_Trans_Ct')
        plt.savefig(
            "./images/eda/bivariate_plots/{}.png".format('Total_Trans_Ct'))
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.savefig(
            "./images/eda/bivariate_plots/correlation.png",
            pad_inches=0.5)
        plt.close(fig)
    def classification_report_image(y_train,
                                    y_test,
                                    y_train_preds_lr,
                                    y_train_preds_rf,
                                    y_test_preds_lr,
                                    y_test_preds_rf):
        """
        Parameters
        ----------
        y_train : pd.Series
            The training labels
        y_test : pd.Series
            The test labels
        y_train_preds_lr : pd.Series
            The training set predictions using Logistic Regression
        y_train_preds_rf : pd.Series
            The training set predictions using Random Forest
        y_test_preds_lr : pd.Series
            The test set predictions using Logistic Regression
        cls_report : dict
            A dictionary of classification report titles as keys and classification 
            report data as values

        Returns
        -------
        None

        Notes
        -----
        To be used for analyzing the classification of the models' parameters passed
        """
        cls_report = {
            "Random Forest": (
                "Random Forest Train",
                classification_report(y_train, y_train_preds_rf),
                "Random Forest Test",
                classification_report(y_test, y_test_preds_rf)),
            "Logistic Regression": (
                "Logistic Regression Train",
                classification_report(y_train, y_train_preds_lr),
                "Logistic Regression Test",
                classification_report(y_test, y_test_preds_lr))
        }
        for title, cls_data in cls_report.items():
            fig = plt.figure(figsize=(20, 10),
                             tight_layout=True)
            plt.rcParams.update({'figure.max_open_warning': 0,
                                 'font.size': 10})
            plt.text(0.01, 1.25, str(cls_data[0]),
                     fontproperties='monospace')
            plt.text(0.01, 0.05, str(cls_data[1]),
                     fontproperties='monospace')
            plt.text(0.01, 0.6, str(cls_data[2]),
                     fontproperties='monospace')
            plt.text(0.01, 0.7, str(cls_data[3]),
                     fontproperties="monospace")
            plt.axis('off')
            plt.savefig("images/results/%s_report.png"%title)
            plt.close(fig)
def feature_importance_plot(model,x_train, output_pth):
    """
    feature_importance_plot(model,x_train, output_pth)
    This function is used in order to plot the feature 
    importance of a given model.
    
    Parameters
    ----------
    model : 
        A trained model.
    x_train : 
        Training data's feature set.
    output_pth :
        Path to save the plot.
    
    Returns
    -------
    None.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_train.columns[i] for i in indices]
    plt.rcParams.update({'figure.max_open_warning': 0})
    fig = plt.figure(figsize=(15, 10))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(x_train.shape[1]), importances[indices])  # Add bars
    plt.xticks(range(x_train.shape[1]), names, rotation=90)
    plt.savefig("%s/feature_Importance.png"%output_pth)
    plt.close(fig)
    feature_csv = pd.DataFrame(list(zip(names, importances)),
                               columns=['Names', 'Feature_Importance'])
    feature_csv.sort_values(by=['Feature_Importance'], ascending=False)
    feature_csv.to_csv("./data/feature_importance_values",
                       float_format='%.4f')  # rounded to four decimals


class Model:
    """
    This class performs a simple classification task.
    
    Parameters
    ----------
    dataset : matrix-like, shape (n_samples, n_features)
        Training data.
    model: str
        The model to be trained.

    Attributes
    ----------
    x_train : array-like, shape = [n_samples, n_features]
        Training set.

    x_test : array-like, shape = [n_samples, n_features]
        Cross-validation set.

    y_train : array-like, shape = [n_samples]
        Training labels
    """

    def __init__(self):
        """
        initialization
        """
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.dataset = None

    def import_dataset(self, pth):
        """
        This function takes in a .csv file path and Registers
        the dataset in a variable called dataset.
        """
        data = pd.read_csv(pth)
        # Flag churn customer
        data['Churn'] = data['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        # Drop column and save as self.dataset
        self.dataset = data.drop(
            ['Unnamed: 0', 'CLIENTNUM', 'Attrition_Flag'], axis=1)
        return self.dataset

    def encoder_helper(self, category_lst):
        """
        This function will take in a dataframe and a list of categorical variables.
        It will then create a new column for each variable in the list, appending
        the string "_Churn" to the original name. The value of each new column
        will be the mean rate of Churn for the respective category.

        Args:
            df (pd.DataFrame): the dataframe to be used
            category_lst (list): a list of categorical variables

        Returns:
            pd.DataFrame: a new dataframe with the new columns
        """
        # create a variable for the categorical columns
        dataframe = self.dataset.copy()
        for cat_name in category_lst:
            category_lst = []
            category_groups = dataframe.groupby(cat_name).mean()["Churn"]
            for val in dataframe[cat_name]:
                category_lst.append(category_groups.loc[val])
            dataframe["%s_%s"%(cat_name, "Churn")] = category_lst
            self.dataset = dataframe
        return self.dataset

    def perform_feature_engineering(self, response):
        """
        Description:
            Perform feature engineering on the dataset, including scaling and train/test split.

        Args:
            response: The feature to be predicted.

        Returns:
            self.x_train: The feature engineering dataset for training.
            self.x_test: The feature engineering dataset for testing.
            self.y_train: The response variable for training.
            self.y_test: The response variable for testing.
        """
        copy_df = self.dataset.copy()
        x_data = copy_df[response]
        y_data = copy_df["Churn"]
        # Standardization
        scaler = StandardScaler()
        scaler_transform = scaler.fit_transform(x_data)
        x_scaled = pd.DataFrame(scaler_transform, columns=x_data.columns)
        # train test split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x_scaled, y_data, test_size=0.3, random_state=42)
        return self.x_train, self.x_test, self.y_train, self.y_test

    def training_models(self):
        """
        This function runs the training models, grid search and saves the models results.
        
        Parameters:
        self: the instance of the class.

        Returns: 
        Saves the best model to a pickle file for later loading and prediction
        """
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
        # set parameters for tuning
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }
        cv_rfc = GridSearchCV(estimator=rfc,
                              param_grid=param_grid,
                              cv=5)
        cv_rfc.fit(self.x_train, self.y_train)
        lrc.fit(self.x_train, self.y_train)
        # choosing the best estimators
        y_train_preds_rf = cv_rfc.best_estimator_.predict(self.x_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(self.x_test)
        y_train_preds_lr = lrc.predict(self.x_train)
        y_test_preds_lr = lrc.predict(self.x_test)
        # save model
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')
        # model report
        Visual.classification_report_image(self.y_train,
                                           self.y_test,
                                           y_train_preds_lr,
                                           y_train_preds_rf,
                                           y_test_preds_lr,
                                           y_test_preds_rf)
        # load model for roc plot
        rfc_model = joblib.load('./models/rfc_model.pkl')
        lrc_model = joblib.load('./models/logistic_model.pkl')
        # feature importance
        feature_importance_plot(cv_rfc.best_estimator_,
                                       self.x_train,
                                       "./images/results/")
        # ROC plots
        fig = plt.figure(figsize=(15, 10), tight_layout=True)
        plt.rcParams.update({'figure.max_open_warning': 0})
        plot_roc_curve(rfc_model,
                       self.x_test,
                       self.y_test,
                       ax=plt.gca(),
                       alpha=0.8)
        plot_roc_curve(lrc_model,
                       self.x_test,
                       self.y_test,
                       ax=plt.gca(),
                       alpha=0.8)
        plt.savefig("./images/results/roc_curve_result.png")
        plt.close(fig)


if __name__ == "__main__":
    # object configuration
    # model object initiation
    MODEL = Model()
    VISUALS = Visual()
    # read the data
    PATH = "./data/bank_data.csv"
    load_data = MODEL.import_dataset(PATH)
    QUANT_COLUMNS = load_data.select_dtypes(include="number")
    CAT_COLUMNS = load_data.select_dtypes(exclude="number")
    # create eda plot and save the result in images/eda
    VISUALS.perform_eda(load_data)
    RESPONSE = keep_columns  # list is in the lists_vars file

    # encoding categorical feature
    MODEL.encoder_helper(CAT_COLUMNS)

    # feature engineering (standardization and data splitting)
    MODEL.perform_feature_engineering(RESPONSE)
    # model training and evaluation
    # model object was saved with .pkl extension in models folder
    # model evaluation result was saved in images/results
    MODEL.training_models()
