'''
Test and Logging for the file churn_library

Author: Emmanuel AKinwale
Date  : April 2022
'''
import time
import logging
import pandas as pd
import churn_library as cls
from lists_vars import keep_columns

# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='./logs/churn_library.log',
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)
def test_import(import_data):
    '''
    test data import - this is the start of the compute time function --start_time
    '''
    import_log = logging.getLogger('test_import')
    import_log.info("Starting import_data")
    try:
        data_set = import_data("./data/bank_data.csv")
        import_log.info("Import_data testing: SUCCESS")
    except FileNotFoundError as err:
        import_log.error("Import_data testing: The file wasn't found")
        raise err
    try:
        assert data_set.shape[0] > 0
        assert data_set.shape[1] > 0
        import_log.info(
            "Test_data is loaded and there are %s rows and %s columns"%(data_set.shape[0], data_set.shape[1]))
    except AssertionError as err:
        import_log.error(
            "Import_data testing: The file path is loaded\
            and the dataset doesn't appear to have rows and columns")
        raise err
    import_log.info("Import test is complete")
def test_eda(import_data):
    '''
    test perform eda function
    '''
    eda_log =logging.getLogger('EDA test')
    eda_log.info("Starting import_data")
    try:
        df=import_data("./data/bank_data.csv")
        eda = cls.Visual()
        eda.perform_eda(df)
        eda_log.info("Perform_EDA testing: SUCCESS")
    except AttributeError as err:
        eda_log.error("Perform_EDA testing: input. data should be a dataframe")
        raise err
    except SyntaxError as err:
        eda_log.error("Perform_EDA testing: input data should be a dataframe")
        raise err
    eda_log.info("EDA test is complete")
def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    encoder_log = logging.getLogger('Encoder helper test')
    encoder_log.info('Starting encoder helper test')
    try:
        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
        encoder_helper(category_lst=cat_columns)
        encoder_log.info("Encoder_helper testing: SUCCESS")
    except KeyError as err:
        logging.error(
            "Encoder_helper testing:\
            There are column names that doesn't exist in your dataframe")
        raise err
    try:
        assert isinstance(cat_columns, list)
        assert len(cat_columns) > 0
    except AssertionError as err:
        encoder_log.error(
            "Encoder_helper testing: category_lst argument should be a list with length > 0")
        raise err
    encoder_log.info("Encoder_helper test complete")
def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    feature_log = logging.getLogger('feature engineering test')
    feature_log.info('Starting feature engineering test')
    try:
        keep_columns = [
            'Customer_Age',
            'Dependent_count',
            'Months_on_book',
            'Total_Relationship_Count',
            'Months_Inactive_12_mon',
            'Contacts_Count_12_mon',
            'Credit_Limit',
            'Total_Revolving_Bal',
            'Avg_Open_To_Buy',
            'Total_Amt_Chng_Q4_Q1',
            'Total_Trans_Amt',
            'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1',
            'Avg_Utilization_Ratio',
            'Gender_Churn',
            'Education_Level_Churn',
            'Marital_Status_Churn',
            'Income_Category_Churn',
            'Card_Category_Churn']
        perform_feature_engineering(response=keep_columns)
    except Exception as err:
        feature_log.error(
            "Perform_feature_engineering testing: Check the main file for variable name")
        raise err
    feature_log.info("Feature engineering test complete")
def test_train_models(training_models):
    '''
    test train_models
    '''
    model_log = logging.getLogger('model training')
    model_log.info('Starting model training')
    try:
        model_train = training_models()
        model_log.info("Training_models testing: SUCCESS")
    except Exception:
        model_log.error(
            "Training_models testing: Out of memory while train the models")
        raise
    model_log.info("Train test complete")
if __name__ == "__main__":
        MODEL = cls.Model()
        VISUALS = cls.Visual()
        test_import(MODEL.import_dataset)
        test_eda(MODEL.import_dataset)
        test_encoder_helper(MODEL.encoder_helper)
        test_perform_feature_engineering(MODEL.perform_feature_engineering)
        test_train_models(MODEL.training_models)
        script_log = logging.getLogger('Ending of all unit tests')
        script.info("Complete")
        