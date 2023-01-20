import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import numpy as np
from uncertainties import ufloat
import itertools as iter

# code link: https://github.com/saidakhursheed/clusteringposter
DATA = pd.read_csv('C:\\Users\\RajaI\\Desktop\\amna\\climate change\\data.csv')


def file(DATA):
    """
    Function to read data and return data (countries as columns and years as
    columns)
    """
    MyYearsData = DATA
    MyCountriesData = MyYearsData.set_index('Country Name').T

    return MyYearsData, MyCountriesData


print(file(DATA))


def Population():
    '''
    Function to analyse changes in population of countries over time using
    line graph
    '''
# Calling function File(DATA) for performing data cleaning
    file(DATA)


# Extract 1990, 2009 data of Countries
    MyCountries = DATA[["Country Name", "Indicator Code", "1990",
                        "2009"]]
    # Urban Population data for All countries
    UrbanPop = MyCountries[((MyCountries['Indicator Code'] == 'SP.URB.TOTL'))]
    TotalPop = MyCountries[((MyCountries['Indicator Code'] == 'SP.POP.TOTL'))]

# Dropping the rows with null values
    UrbanPop = UrbanPop.dropna()
    TotalPop = TotalPop.dropna()
# Reset Index of Dataframe
    Population.UrbanPop = UrbanPop.reset_index(drop=True)
    # print(UrbanPop)
    Population.TotalPop = TotalPop.reset_index(drop=True)


Population()

UrbanPop = Population.UrbanPop
TotalPop = Population.TotalPop


#Normalization
scaler = MinMaxScaler()
scaled_UrbanPop = scaler.fit_transform(UrbanPop.iloc[:, -2:])
scaled_TotalPop = scaler.fit_transform(TotalPop.iloc[:, -2:])


# Use KMeans to find 3 clusters in the data
kmeans = KMeans(n_clusters=3)
y_kmeans = kmeans.fit_predict(scaled_UrbanPop)

# Add the cluster as a new column to the dataframe
UrbanPop['cluster1'] = kmeans.labels_


# create a plot showing the clusters and cluster centers using pyplot
plt.scatter(scaled_UrbanPop[y_kmeans == 0, 0], scaled_UrbanPop[y_kmeans == 0, 1], c='red',
            label='Cluster 1')
plt.scatter(scaled_UrbanPop[y_kmeans == 1, 0], scaled_UrbanPop[y_kmeans == 1, 1], c='green',
            label='Cluster 2')
plt.scatter(scaled_UrbanPop[y_kmeans == 2, 0], scaled_UrbanPop[y_kmeans == 2, 1], c='blue',
            label='Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('1990')
plt.ylabel('2019')
plt.title('Cluster Membership and Centers of Urban Population')
plt.legend()
plt.show()


# Use KMeans to find 5 clusters in the data
kmeans = KMeans(n_clusters=3)
y_kmeans = kmeans.fit_predict(scaled_TotalPop)

# Add the cluster as a new column to the dataframe
TotalPop['cluster1'] = kmeans.labels_


# create a plot showing the clusters and cluster centers using pyplot
plt.scatter(scaled_TotalPop[y_kmeans == 0, 0], scaled_TotalPop[y_kmeans == 0, 1], c='red',
            label='Cluster 1')
plt.scatter(scaled_TotalPop[y_kmeans == 1, 0], scaled_TotalPop[y_kmeans == 1, 1], c='green',
            label='Cluster 2')
plt.scatter(scaled_TotalPop[y_kmeans == 2, 0], scaled_TotalPop[y_kmeans == 2, 1], c='blue',
            label='Cluster 3')


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('1990')
plt.ylabel('2019')
plt.title('Cluster Membership and Centers of Total Population')
plt.legend()
plt.show()

DATA = DATA[(DATA['Country Name'] == 'Aruba')]
mydata = DATA[((DATA['Indicator Code'] == 'SP.URB.TOTL'))]
print(mydata)
mydata = pd.melt(mydata, id_vars="Country Name",
               value_vars= ['1996', '1997', '1998', '1999', '2000', '2001',
                           '2002', '2003', '2004', '2005', '2006', '2007',
                           '2008', '2009', '2010', '2011', '2012', '2013',
                           '2014', '2015', '2016', '2017', '2018', '2019'
                           ], var_name="Years",value_name="Urban_Population")


mydata["Years"] = mydata["Years"].astype(float)


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper


def logistics(t, n0, g, t0):
    """Calculates the logistic function" """
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f


# Use err_ranges function to estimate lower and upper limits
x = mydata['Years']
y = mydata['Urban_Population']

param, covar = curve_fit(logistics, x, y)
mydata["log"] = logistics(mydata['Years'], *param)
plt.plot(x, y, label="data")
plt.plot(x, mydata["log"], label="fit")
plt.legend()
plt.title("First attempt of curve fitting")
plt.xlabel("year")
plt.ylabel("Urban Pop of Aruba")
plt.show()
print()


# estimated turning year: 2014
param = [44351, 0.02, 2014]
mydata["log"] = logistics(x, *param)
plt.plot(x, y, label="data")
plt.plot(x, mydata["log"], label="fit")
plt.legend()
plt.xlabel("year")
plt.ylabel("Urban Pop of Aruba")
plt.title("Start value of Curve fit")
plt.show()


param, covar = curve_fit(logistics, x, y,
                         p0=[44351, 0.02, 2014])
print("Fit parameter", param)
mydata["log"] = logistics(x, *param)
sigma = np.sqrt(np.diagonal(covar))
a = ufloat(param[0], sigma[0])
b = ufloat(param[1], sigma[1])
x_pred = np.linspace(1995, 2028, 20)
plt.plot(x, y, label="Original data")
plt.plot(x_pred, logistics(x_pred, *param), 'red', label="fit")
upper = logistics(x_pred, *(param + sigma))
lower = logistics(x_pred, *(param - sigma))
# plotting the confidence intervals
plt.fill_between(x_pred, lower, upper,
                 color='green', alpha=0.15, label="Confidence Interval")
plt.legend()
plt.title("Final curve fitting")
plt.xlabel("year")
plt.ylabel("Urban Pop of Aruba")
plt.show()


