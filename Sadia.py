import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:\\Users\\RajaI\\Desktop\\climate change\\data.csv')


def file(df):
    """
    Function to read data and return data (countries as columns and years as
    columns)
    """
    MyYearsData = df
    MyCountriesData = MyYearsData.set_index('Country Name').T

    return MyYearsData, MyCountriesData


print(file(df))


def Population():
    '''
    Function to analyse changes in population of countries over time using
    line graph
    '''
# Calling function Read_File(DATA) for performing data cleaning
    Read_File(DATA)

# Clean up Dataset for relavant information
# Extract 1990, 2009 and 2019 data of Countries
    MyCountries = DATA[["Country Name", "Indicator Code", "1990",
                        "2009", "2019"]]
    # Urban Population data for 8 countries
    UrbanPop = MyCountries[((MyCountries['Indicator Code'] == 'SP.URB.TOTL')) &
                           ((MyCountries['Country Name'] == 'China') |
                           (MyCountries['Country Name'] == 'United States')
                           | (MyCountries['Country Name'] == 'Germany')
                           | (MyCountries['Country Name'] == 'Japan')
                           | (MyCountries['Country Name'] == 'India')
                           | (MyCountries['Country Name'] == 'Aruba') |
                           (MyCountries['Country Name'] == 'South Africa')
                           | (MyCountries['Country Name'] == 'Zambia'))]


# Dropping the rows with null values
    UrbanPop = UrbanPop.dropna()
# Reset Index of Dataframe
    UrbanPop = UrbanPop.reset_index(drop=True)
    # print(UrbanPop)

    # Total Population data for 8 countries
    TotalPop = MyCountries[((MyCountries['Indicator Code'] == 'SP.POP.TOTL')) &
                           ((MyCountries['Country Name'] == 'China') |
                           (MyCountries['Country Name'] == 'United States')
                           | (MyCountries['Country Name'] == 'Germany')
                           | (MyCountries['Country Name'] == 'Japan')
                           | (MyCountries['Country Name'] == 'India')
                           | (MyCountries['Country Name'] == 'Aruba') |
                           (MyCountries['Country Name'] == 'South Africa')
                           | (MyCountries['Country Name'] == 'Zambia'))]


# Dropping the rows with null values
    TotalPop = TotalPop.dropna()
# Reset Index of Dataframe
    TotalPop = TotalPop.reset_index(drop=True)
    # print(UrbanPop)

    fig, (ax, ax2) = plt.subplots(ncols=2, sharey=True)
    ax.invert_xaxis()
    ax.yaxis.tick_right()

    TotalPop.plot(title='Total Population of Countries',
                  x="Country Name", kind="bar", legend=False, ax=ax)
    UrbanPop.plot(title='Urban Population of Countries',
                  x="Country Name", kind="bar", ax=ax2)
    plt.tight_layout()
    plt.show()


Population()
