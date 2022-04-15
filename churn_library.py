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
    Visual class perform eda, classification report and csv,
    the feature importances figure on df,
    and save figures to images folder --all data visuals
    input:
        df: pandas dataframe
    output:
        None
    '''

    def __init__(self):
        """
        initialization
        self.output_pth:
            path to store the feature_importance plots/figures
        """

    # plot all columns in the Quant_cols anmd Categorical_Cols list from the
    # dataframe (df)
    def perform_eda(self, df):
        """
        perform eda on df and save figures to images folder
        input:
            dataframe: pandas dataframe
        output:
            None
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
        '''
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
        output:
            None
        '''
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
    '''
    creates and stores the feature importances in pth
    input:
        self.model: model object containing feature_importances_
        X_data: pandas dataframe of X values
        self.output_pth: path to store the figure
    output:
        None
    '''
    # compute feature importances
    # Sort feature importances in descending order
    # Rearrange feature names so they match the sorted feature importances
    # plot feature importances
    # Add feature names as x-axis labels
    # store importance values in a csv
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
                       float_format='%.4f')  # rounded to two decimals


class Model:
    """
    Model class for the following functions
    import_dataset: loads the data
    encoder_helper: one_hot_encoder for categorical columns
    perform_feature_engineering: standardization of x_train to reduce model overfitting
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
        '''
        returns dataframe for the csv found at pth
        input:
            pth: a path to the csv
        output:
            df: pandas dataframe
        '''
        data = pd.read_csv(pth)
        # Flag churn customer
        data['Churn'] = data['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        # Drop column and save as self.dataset
        self.dataset = data.drop(
            ['Unnamed: 0', 'CLIENTNUM', 'Attrition_Flag'], axis=1)
        return self.dataset

    def encoder_helper(self, category_lst):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category
        input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
        output:
            df: pandas dataframe with new columns for
        '''
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
        '''
        input:
            df: pandas dataframe
        response: string of response name [optional argument that could be used for naming variables
        or index y column]
        output:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data
        '''
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
        '''
        train, store model results: images + scores, and store models
        input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
        output:
              None
        '''
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
