from datetime import datetime
import pandas
from application_logging.logger import App_Logger
import os
from os import listdir


class dataTransformPredict:

     """
                  This class shall be used for transforming the Good Raw Training Data before loading it in Database!!.



                  """

     def __init__(self):
          self.goodDataPath = "Prediction_Raw_Files_Validated/Good_Raw"
          self.logger = App_Logger()


     def replaceMissingWithNull(self):

          """
                                  Method Name: replaceMissingWithNull
                                  Description: This method replaces the missing values in columns with "NULL" to
                                               store in the table. We are using substring in the first column to
                                               keep only "Integer" data for ease up the loading.
                                               This column is anyways going to be removed during prediction.


                                          """

          try:
               log_file = "Prediction_Logs/dataTransformLog.txt"
               onlyfiles = [f for f in listdir(self.goodDataPath)]
               for file in onlyfiles:
                    data = pandas.read_csv(self.goodDataPath + "/" + file)
                    # list of columns with string datatype variables
                    columns = ["policy_bind_date","policy_state","policy_csl","insured_sex","insured_education_level","insured_occupation","insured_hobbies","insured_relationship","incident_state","incident_date","incident_type","collision_type","incident_severity","authorities_contacted","incident_city","incident_location","property_damage","police_report_available","auto_make","auto_model"]

                    for col in columns:
                         data[col] = data[col].apply(lambda x: "'" + str(x) + "'")
                    # #csv.update("'"+ csv['Wafer'] +"'")
                    # csv.update(csv['Wafer'].astype(str))
                    #csv['Wafer'] = csv['Wafer'].str[6:]
                    data.to_csv(self.goodDataPath+ "/" + file, index=None, header=True)
                    self.logger.log(log_file," %s: File Transformed successfully!!" % file)
               #log_file.write("Current Date :: %s" %date +"\t" + "Current time:: %s" % current_time + "\t \t" +  + "\n")

          except Exception as e:
               self.logger.log(log_file, "Data Transformation failed because:: %s" % e)
               #log_file.write("Current Date :: %s" %date +"\t" +"Current time:: %s" % current_time + "\t \t" + "Data Transformation failed because:: %s" % e + "\n")
               # log_file.close()
               raise e
          # log_file.close()
