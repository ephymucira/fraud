from datetime import datetime
from Training_Raw_data_validation.rawValidation import Raw_Data_validation
from DataTypeValidation_Insertion_Training.DataTypeValidation import dBOperation
from DataTransform_Training.DataTransformation import dataTransform
from application_logging import logger
import os

class train_validation:
    def __init__(self,path,log_file):
        # self.raw_data = Raw_Data_validation(path)
        self.dataTransform = dataTransform()
        self.dBOperation = dBOperation()
        self.log_file = log_file
        # self.file_object = open(log_file, 'a+')
        # self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        # self.raw_data = Raw_Data_validation(path,self.file_object)
        self.raw_data = Raw_Data_validation(path,log_file)
        self.log_writer = logger.App_Logger()

    def train_validation(self):
        try:
            self.log_writer.log(self.log_file, 'Start of Validation on files for Training!!')
            # extracting values from prediction schema
            LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, noofcolumns = self.raw_data.valuesFromSchema()
            # getting the regex defined to validate filename
            regex = self.raw_data.manualRegexCreation()
            # validating filename of prediction files
            self.raw_data.validationFileNameRaw(regex, LengthOfDateStampInFile, LengthOfTimeStampInFile)
            # validating column length in the file
            self.raw_data.validateColumnLength(noofcolumns)
            # validating if any column has all values missing
            self.raw_data.validateMissingValuesInWholeColumn()
            self.log_writer.log(self.log_file, "Raw Data Validation Complete!!")

            self.log_writer.log(self.log_file, "Starting Data Transforamtion!!")
            # replacing blanks in the csv file with "Null" values to insert in table
            self.dataTransform.replaceMissingWithNull()

            self.log_writer.log(self.log_file, "DataTransformation Completed!!!")

            self.log_writer.log(self.log_file,
                                "Creating Training_Database and tables on the basis of given schema!!!")
            # create database with given name, if present open the connection! Create table with columns given in schema
            self.dBOperation.createTableDb('Training', column_names)
            self.log_writer.log(self.log_file, "Table creation Completed!!")
            self.log_writer.log(self.log_file, "Insertion of Data into Table started!!!!")
            # insert csv files in the table
            self.dBOperation.insertIntoTableGoodData('Training')
            self.log_writer.log(self.log_file, "Insertion in Table completed!!!")
            self.log_writer.log(self.log_file, "Deleting Good Data Folder!!!")
            # Delete the good data folder after loading files in table
            self.raw_data.deleteExistingGoodDataTrainingFolder()
            self.log_writer.log(self.log_file, "Good_Data folder deleted!!!")
            self.log_writer.log(self.log_file, "Moving bad files to Archive and deleting Bad_Data folder!!!")
            # Move the bad files to archive folder
            self.raw_data.moveBadFilesToArchiveBad()
            self.log_writer.log(self.log_file, "Bad files moved to archive!! Bad folder Deleted!!")
            self.log_writer.log(self.log_file, "Validation Operation completed!!")
            self.log_writer.log(self.log_file, "Extracting csv file from table")
            # export data in table to csvfile
            self.dBOperation.selectingDatafromtableintocsv('Training')
            # self.log_file.close()

        except Exception as e:
            raise e









