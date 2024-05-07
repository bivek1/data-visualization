# %%
import pandas as pd

# Load the data into a DataFrame
df = pd.read_csv('stock.csv')

# 1. Total number of rows and columns
total_rows, total_columns = df.shape
print("Total rows:", total_rows)
print("Total columns:", total_columns)

# 2. Check for null values and calculate total null values
null_values = df.isnull().sum().sum()
print("Total null values:", null_values)

# Alternatively, you can check for null values per column
null_values_per_column = df.isnull().sum()
print("Null values per column:")
print(null_values_per_column)

# %%
# Sample DataFrame
# Calculate IQR
q1 = df['Open'].quantile(0.25)
q3 = df['Close'].quantile(0.75)
iqr = q3 - q1

# Define outlier thresholds (e.g., 1.5 times IQR)
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Identify potential outliers
outliers = (df['Open'] < lower_bound) | (df['Close'] > upper_bound)
print(df[outliers])

# %%
# 1. Data processing 1 Remove the 'Stock Splits' column
df.drop('Stock Splits', axis=1, inplace=True)
# 2. Data processing Remove the 'Divident' column
df.drop('Dividends', axis=1, inplace=True)

# %%

# 3. Round the 'Open', 'High', 'Low', and 'Close' columns to 4 decimal places
df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].round(4)
#4. Change date time format and remove time due to same time in all data 
df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.date

# Display the first few rows of the DataFrame to verify the changes
print(df.head())

# %%
# 5. Create a dictionary mapping short forms to full forms
companies = {
    'AAPL': {'Company': 'Apple Inc.', 'Category': 'Technology'},
    'MSFT': {'Company': 'Microsoft Corporation', 'Category': 'Technology'},
    'GOOGL': {'Company': 'Alphabet Inc. (formerly Google Inc.)', 'Category': 'Technology'},
    'AMZN': {'Company': 'Amazon.com Inc.', 'Category': 'Technology'},
    'NVDA': {'Company': 'NVIDIA Corporation', 'Category': 'Technology'},
    'META': {'Company': 'Meta Platforms Inc. (formerly Facebook, Inc.)', 'Category': 'Technology'},
    'TSLA': {'Company': 'Tesla, Inc.', 'Category': 'Technology'},
    'LLY': {'Company': 'Eli Lilly and Company', 'Category': 'Healthcare'},
    'V': {'Company': 'Visa Inc.', 'Category': 'Financial Services'},
    'TSM': {'Company': 'Taiwan Semiconductor Manufacturing Company Limited', 'Category': 'Technology'},
    'UNH': {'Company': 'UnitedHealth Group Incorporated', 'Category': 'Healthcare'},
    'AVGO': {'Company': 'Broadcom Inc.', 'Category': 'Technology'},
    'NVO': {'Company': 'Novo Nordisk A/S', 'Category': 'Healthcare'},
    'JPM': {'Company': 'JPMorgan Chase & Co.', 'Category': 'Financial Services'},
    'WMT': {'Company': 'Walmart Inc.', 'Category': 'Retail'},
    'XOM': {'Company': 'Exxon Mobil Corporation', 'Category': 'Energy'},
    'MA': {'Company': 'Mastercard Incorporated', 'Category': 'Financial Services'},
    'JNJ': {'Company': 'Johnson & Johnson', 'Category': 'Healthcare'},
    'PG': {'Company': 'Procter & Gamble Company', 'Category': 'Consumer Goods'},
    'ORCL': {'Company': 'Oracle Corporation', 'Category': 'Technology'},
    'HD': {'Company': 'The Home Depot, Inc.', 'Category': 'Retail'},
    'ADBE': {'Company': 'Adobe Inc.', 'Category': 'Technology'},
    'ASML': {'Company': 'ASML Holding N.V.', 'Category': 'Technology'},
    'CVX': {'Company': 'Chevron Corporation', 'Category': 'Energy'},
    'COST': {'Company': 'Costco Wholesale Corporation', 'Category': 'Retail'},
    'TM': {'Company': 'Toyota Motor Corporation', 'Category': 'Automotive'},
    'MRK': {'Company': 'Merck & Co., Inc.', 'Category': 'Healthcare'},
    'KO': {'Company': 'The Coca-Cola Company', 'Category': 'Consumer Goods'},
    'ABBV': {'Company': 'AbbVie Inc.', 'Category': 'Healthcare'},
    'BAC': {'Company': 'Bank of America Corporation', 'Category': 'Financial Services'},
    'PEP': {'Company': 'PepsiCo, Inc.', 'Category': 'Consumer Goods'},
    'FMX': {'Company': 'Fomento Economico Mexicano, S.A.B. de C.V.', 'Category': 'Consumer Goods'},
    'CRM': {'Company': 'Salesforce.com, Inc.', 'Category': 'Technology'},
    'SHEL': {'Company': 'Royal Dutch Shell plc', 'Category': 'Energy'},
    'ACN': {'Company': 'Accenture plc', 'Category': 'Technology'},
    'NFLX': {'Company': 'Netflix, Inc.', 'Category': 'Technology'},
    'MCD': {'Company': "McDonald's Corporation", 'Category': 'Consumer Goods'},
    'AMD': {'Company': 'Advanced Micro Devices, Inc.', 'Category': 'Technology'},
    'LIN': {'Company': 'Linde plc', 'Category': 'Technology'},
    'NVS': {'Company': 'Novartis AG', 'Category': 'Healthcare'},
    'AZN': {'Company': 'AstraZeneca PLC', 'Category': 'Healthcare'},
    'CSCO': {'Company': 'Cisco Systems, Inc.', 'Category': 'Technology'},
    'TMO': {'Company': 'Thermo Fisher Scientific Inc.', 'Category': 'Healthcare'},
    'BABA': {'Company': 'Alibaba Group Holding Limited', 'Category': 'Technology'},
    'INTC': {'Company': 'Intel Corporation', 'Category': 'Technology'},
    'PDD': {'Company': 'Pinduoduo Inc.', 'Category': 'Technology'},
    'SAP': {'Company': 'SAP SE', 'Category': 'Technology'},
    'ABT': {'Company': 'Abbott Laboratories', 'Category': 'Healthcare'},
    'TMUS': {'Company': 'T-Mobile US, Inc.', 'Category': 'Telecommunications'},
    'PFE': {'Company': 'Pfizer Inc.', 'Category': 'Healthcare'},
    'AFL': {'Company': 'Aflac Incorporated', 'Category': 'Financial Services'},
    'JPM': {'Company': 'JPMorgan Chase & Co.', 'Category': 'Financial Services'},
    'BAC': {'Company': 'Bank of America Corporation', 'Category': 'Financial Services'},
}


#4 Filter the DataFrame based on the short forms and replace the "Company" column values
df = df[df['Company'].isin(companies.keys())]
# df['Company'] = df['Company'].map(companies)
# df['Category'] = df['Company'].apply(lambda x: x['Category'])
# # Display the DataFrame
df['Company'] = df['Company'].map({k: v['Company'] for k, v in companies.items()})

# Adding Category column
df['Category'] = df['Company'].map({v['Company']: v['Category'] for v in companies.values()})

# Display the DataFrame
print(df)
print(df.head)


# %%
# 6. Reset index starting from 1
import numpy as np

df.index = np.arange(1, len(df) + 1)

# Print the DataFrame
print(df)


# %%
#Data preprocessing 6 Add Seasonal category based on date 
# Define a function to map month to season
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8, 9]:
        return 'Summer'
    else:
        return 'Fall'

df['Month'] = df['Date'].apply(lambda x: x.month)
df['Season'] = df['Month'].apply(get_season)

# Drop the 'Month' column if not needed anymore
df.drop(columns=['Month'], inplace=True)

print(df)

# %%
# 2. Check for null values and calculate total null values
null_values = df.isnull().sum().sum()
print("Total null values:", null_values)

# Alternatively, you can check for null values per column
null_values_per_column = df.isnull().sum()
print("Null values per column:")
print(null_values_per_column)

print(df)

# %%
total_rows, total_columns = df.shape
print("Total rows:", total_rows)
print("Total columns:", total_columns)

# %%
print(df['Company'])

# %%
print(df.head)

# %%
import pandas as pd

# Assuming 'df' is your DataFrame containing a 'Company' column (potentially with duplicates)

# 1. Get total number of companies in each category
category_counts = df['Category'].value_counts()
print(category_counts, end="\n\n")  # Print with a newline for clarity

# 2. Get all unique companies (regardless of category)
unique_companies = df['Company'].nunique()  # Use nunique() for unique count
print(f"Total Unique Companies: {unique_companies}")

# %%
import matplotlib.pyplot as plt

#1 Group the DataFrame by the "Category" column and sum the closing share prices
category_totals = df.groupby('Category')['Close'].sum()

# Plot the total closing share prices for each category
plt.figure(figsize=(10, 6))
category_totals.plot(kind='bar', color='skyblue')
plt.title('Total Closing Share Prices by Category')
plt.xlabel('Category')
plt.ylabel('Total Closing Share Price')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import pandas as pd

# Convert 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Get unique companies
technology_companies = df[df['Category'] == 'Technology']['Company'].unique()[:10]

# Plot closing prices over time for each company
plt.figure(figsize=(12, 8))
for company in technology_companies:
    company_data = df[df['Company'] == company]
    plt.plot(company_data['Date'], company_data['Close'], label=company)

plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Closing Prices Over Time for Each Company')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# %%
import matplotlib.pyplot as plt
import pandas as pd

# Convert 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Get unique companies
technology_companies = df[df['Category'] == 'Healthcare']['Company'].unique()[:10]

# Plot closing prices over time for each company
plt.figure(figsize=(12, 8))
for company in technology_companies:
    company_data = df[df['Company'] == company]
    plt.plot(company_data['Date'], company_data['Close'], label=company)

plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Closing Prices Over Time for Each Company')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# %%
import matplotlib.pyplot as plt
import pandas as pd

# Convert 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Filter technology and non-technology companies
# Calculate total share values for technology and non-technology companies
category_total_share = df.groupby('Category')['Close'].sum()

# Plot the pie chart
plt.figure(figsize=(8, 8))
labels = category_total_share.index
sizes = category_total_share.values
colors = plt.cm.tab20.colors  # Using a colormap for colors
explode = [0.1] * len(labels)  # explode all slices
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Total Share Values by Category')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np

# Assuming df is your DataFrame containing the data
# If not, replace it with your DataFrame name

# Extracting year and high/low stock values from DataFrame
years = pd.to_datetime(df['Date']).dt.year
high_values = df.groupby(years)['Close'].max()
low_values = df.groupby(years)['Close'].min()

fig, ax = plt.subplots()

# Plotting high and low stock values
ax.plot(high_values.index, high_values, color='black', label='High Value')
ax.plot(low_values.index, low_values, color='blue', label='Low Value')

# Highlighting regions where high value > low value and vice versa
ax.fill_between(high_values.index, high_values, where=high_values > low_values, facecolor='green', alpha=.5, label='High > Low')
ax.fill_between(low_values.index, low_values, where=low_values > high_values, facecolor='red', alpha=.5, label='Low > High')

# Adding horizontal line at y=0
ax.axhline(0, color='black')

# Adding legend and labels
ax.legend()
ax.set_xlabel('Year')
ax.set_ylabel('Stock Value')
ax.set_title('High and Low Stock Values Over Time')

plt.show()


# %%
season_counts = df['Season'].value_counts()

# Print the counts
print("Number of data points in each season:")
print(season_counts)

# %%
import matplotlib.pyplot as plt

# Define the years
years = df['Date'].dt.year.unique()

# Define the seasons
seasons = df['Season'].unique()

# Initialize a dictionary to store population data by season
population_by_season = {season: [] for season in seasons}

# Calculate the total trading volume for each season in each year
for year in years:
    year_data = df[df['Date'].dt.year == year]
    for season in seasons:
        season_data = year_data[year_data['Season'] == season]
        total_volume = season_data['Volume'].sum()
        population_by_season[season].append(total_volume)

# Plot the stacked area plot
fig, ax = plt.subplots()
ax.stackplot(years, population_by_season.values(),
             labels=population_by_season.keys(), alpha=0.8)
ax.legend(loc='upper left', reverse=True)
ax.set_title('Seasonal Trading Volume')
ax.set_xlabel('Year')
ax.set_ylabel('Trading Volume')

plt.show()


# %%
df['High'].max()
df['Low'].max()

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = {
    'Date': pd.date_range(start='2018-01-01', end='2023-12-31', freq='D'),
    'High': np.random.rand(730) * 100,  # High values
    'Low': np.random.rand(730) * 50,     # Low values
}


# Extract year and month
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Assign season based on month
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['Season'] = df['Month'].apply(get_season)

# Group by season and year, calculate mean of high and low values
seasonal_data = df.groupby(['Season', 'Year']).agg({'High': 'mean', 'Low': 'mean'}).reset_index()

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
for season, data in seasonal_data.groupby('Season'):
    ax.plot(data['Year'], data['High'], label=f'{season} High')
    ax.plot(data['Year'], data['Low'], label=f'{season} Low')

ax.set_xlabel('Year')
ax.set_ylabel('Stock Value')
ax.set_title('High and Low Stock Values Over Time by Season')
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np

def survey(df):
    """
    Parameters
    ----------
    df : DataFrame
        DataFrame containing the data. Each row represents a category, 
        and each column represents the share value for each year.
    """
    # Filter the DataFrame for the "Technology" category
    df_technology = df[df['Category'] == 'Technology']
    
    # Group by year and category and calculate the mean share value
    grouped = df_technology.groupby('Year')['Close'].mean()

    labels = grouped.index
    data = grouped.values
    category_colors = plt.cm.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, len(data)))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data))

    # Plot the data
    for i, (year, share_value) in enumerate(zip(labels, data)):
        ax.barh(year, share_value, color=category_colors[i])
        ax.text(share_value, year, f'{share_value:.2f}', ha='left', va='center', color='black')

    ax.set_xlabel('Mean Share Value')
    ax.set_ylabel('Year')
    ax.set_title('Mean Share Value of Technology Category Over Years')

    plt.show()

survey(df)



# %%
import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with the necessary columns (Date, Company, Close, etc.)

# Step 1: Calculate the total share value for each company
company_share = df.groupby('Company')['Close'].sum()

# Step 2: Find the company with the highest total share value
top_company = company_share.idxmax()  # Get the index of the company with the highest total share value

# Step 3: Filter the DataFrame for the top performing company and plot its performance over time
top_company_data = df[df['Company'] == top_company]
plt.plot(top_company_data['Date'], top_company_data['Close'])
plt.title(f"Performance of {top_company} Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# %%
# Assuming df is your DataFrame containing company names and categories

# Step 1: Extract unique company names
unique_companies = df['Company'].unique()

# Step 2: For each unique company name, extract the corresponding category
categories = []
for company in unique_companies:
    category = df[df['Company'] == company]['Category'].iloc[0]  # Get the category for the first occurrence of the company
    categories.append(category)

# Step 3: Count the occurrences of each category
category_counts = {}
for category in categories:
    if category in category_counts:
        category_counts[category] += 1
    else:
        category_counts[category] = 1

print(category_counts)



