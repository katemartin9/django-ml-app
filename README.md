# Automating Data Science

This django project has multiple apps:

* regml - regression problem
* classml - classification problem
* clustml - clustering problem
* superml - deep learning problem

The end goal of this web app is to be able to analyse supplied dataset and recommend the best ML model out of the most commonly used. It would all depend on your ML problem.
<br><br>
This tool will perform any data preprocessing needed - data cleaning, feature extraction, normalisation, etc etc. It will visualise the data and look at the relationship between the features. It will be able to deal with numeric, categorical and datetime features separately with very little input from the end user.<br><br>

Doesn't this sound exciting?<br><br>

## REGML - Regression ML

This app has been designed to help data scientists analyse regression datasets and recommend the best ML models.<br>
The data should be supplied in a csv/txt format and there is no limit on the number of columns or their formats. It accepts numeric, categorical or data column types. 

```text
Please note that the quality of the analysis is as good as the data supplied.
The tool will still do its best at identifying the issues and dealing with them.
```

Other than a valid file you need to come up with a project name under 50 characters.
You have an option of removing missing rows. If you leave it unticked the tool will fill all missing values with the median value for numeric columns and the most common value for categorical and date columns. 
The columns that have a very high number of missing values will be dropped.

- [x] Supply a valid file (csv/txt)
- [x] Decide on the missing values
- [x] Choose a project name
- [x] Hit 'Upload'

Once you've hit upload, you will be taken to a data preview page where you can see the first 5 rows of your data. Check that all is looking good.
Next you are asked to check whether the tool identified your column types correctly. Please have a quick look and use the drop-downs to make changes. 
Don't forget to tick your target (Y) value. Then click 'OK' at the bottom of the page.

- [x] Check the data
- [x] Check the column types
- [x] Select your target column (Y)
- [x] Hit 'OK'
 
## CLASSML - Classification ML

Description is yet to come...

## CLUSTML - Clustering ML

Description is yet to come..

## SUPERML - Deep Learning

Description is yet to come...