from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#earth_surface_temperatures.csv
#sample_ml.csv

def Task1():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/earth_surface_temperatures.csv'
    df = pd.read_csv(file_path)
    total = df.isnull().sum()
    return total

def Task1_fill_missing_value():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/earth_surface_temperatures.csv'
    df = pd.read_csv(file_path)
    fill_m = df['Monthly_variation'].mean().round(2)
    fill_a = df['Anomaly'].mean().round()
    df['Monthly_variation'].fillna(fill_m, inplace=True)
    df['Anomaly'].fillna(fill_a, inplace=True)
    time_series = df['Temperature'].median()
    df['Temperature'].fillna(time_series, inplace=True)
    df.to_csv(file_path, index=False)

def Task2_date():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/earth_surface_temperatures.csv'
    df = pd.read_csv(file_path)
    if "Date" not in df.columns:
        df['Date'] = df['Month'].astype(str) + '-' + df['Years'].astype(str)
        df.to_csv(file_path, index=False)

def Task3_Outliers():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/earth_surface_temperatures.csv'
    df = pd.read_csv(file_path)
    df['z_score'] = stats.zscore(df['Temperature'])
    outliers = df[df['z_score'].abs() > 3].round(2)
    if "z_score" not in df.index:
        df['z_score'] = df['z_score']
        df.to_csv(file_path, index=False)
        return outliers
    else:
        print("no outliers")
        return outliers

def Task4_Forall():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/earth_surface_temperatures.csv'
    df = pd.read_csv(file_path)
    finalval = df[['Temperature', 'Monthly_variation', 'Anomaly']].agg(['mean', 'median', 'std', 'min', 'max'])
    print(finalval)

def Task5_Average():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/earth_surface_temperatures.csv'
    df = pd.read_csv(file_path)
    finalaggl =  df.groupby('Country')['Temperature'].mean()
    print(finalaggl)

def Task6_plot_tempvsyear():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/earth_surface_temperatures.csv'
    df = pd.read_csv(file_path)
    finalaggl =  df.groupby('Years')['Temperature'].mean()
    finalaggl.plot(kind='line')
    plt.show()

def Task7_countrytemp():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/earth_surface_temperatures.csv'
    df = pd.read_csv(file_path)
    finalaggl =  df.groupby(['Country','Month'])['Temperature'].agg(['min', 'max'])
    finalaggl.plot(kind='bar')
    plt.show()

def Task8_TemperaturevsAnomalies():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/earth_surface_temperatures.csv'
    df = pd.read_csv(file_path)
    finalaggl =  df.groupby('Month')['Anomaly'].agg(['mean']).unstack().plot(kind='line')
    plt.show()

def Task9_TemperatureCompare():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/earth_surface_temperatures.csv'
    df = pd.read_csv(file_path)
    contrylist = df['Country'].unique().tolist()[:5]
    top5 = df[df['Country'].isin(contrylist)]
    top5.groupby(['Country','Month'])['Temperature'].agg(['mean']).unstack().plot(kind='bar')
    plt.show()

def Task10_correlation():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/earth_surface_temperatures.csv'
    df = pd.read_csv(file_path)
    correlation = df[['Temperature', 'Monthly_variation', 'Anomaly']].corr()
    correlation.plot(kind='scatter', x='Temperature',y='Monthly_variation')
    plt.show()

def Task11_finalplot():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/earth_surface_temperatures.csv'
    df = pd.read_csv(file_path)
    correlation = df[['Month', 'Temperature', 'Monthly_variation']].corr()
    sns.heatmap(correlation.corr(), annot=True)
    plt.show()

#Execution startes here
Task1()
Task1_fill_missing_value()
Task2_date()
Task3_Outliers()
Task4_Forall()
Task5_Average()
Task6_plot_tempvsyear()
Task7_countrytemp()
Task8_TemperaturevsAnomalies()
Task9_TemperatureCompare()
Task10_correlation()
Task11_finalplot()