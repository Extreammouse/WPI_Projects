import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import re

a = 'transaction_data.csv'
b = 'purchase_behaviour.csv'
c = 'transactionPurchase_behaviour_data.csv'

def change_date_transition():
    file_path = "/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/" + a
    df = pd.read_csv(file_path)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.to_csv(file_path, index=False)

def mergecsv():
    file_path = "/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/" + a
    file_patha = "/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/" + b
    df1 = pd.read_csv(file_path)
    df2 = pd.read_csv(file_patha)
    df = pd.merge(df1, df2, on="LYLTY_CARD_NBR")
    df.to_csv('/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/transactionPurchase_behaviour_data.csv', index=False)
    print(df.head())

def fill_missingdata():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/transactionPurchase_behaviour_data.csv'
    output_file = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/processed_data.csv'
    listl = ["MIDAGE SINGLES/COUPLES", "NEW FAMILIES", "OLDER FAMILIES",
             "OLDER SINGLES/COUPLES", "RETIREES", "YOUNG FAMILIES",
             "YOUNG SINGLES/COUPLES"]
    list2 =["Budget","Mainstream","Premium"]

    index = 0
    index2 = 0
    chunk_size = 10000
    if os.path.exists(output_file):
        os.remove(output_file)
    chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
    for chunk in chunk_iter:
        missing_indices = chunk[(chunk['LIFESTAGE'].isna() | (chunk['LIFESTAGE'] == '') | chunk['PREMIUM_CUSTOMER'].isna() | (chunk['PREMIUM_CUSTOMER'] == ''))].index
        for i in missing_indices:
            if pd.isna(chunk.loc[i, 'LIFESTAGE']) or chunk.loc[i, 'LIFESTAGE'] == '':
                chunk.loc[i, 'LIFESTAGE'] = listl[index]
                index = (index + 1) % len(listl)

            if pd.isna(chunk.loc[i, 'PREMIUM_CUSTOMER']) or chunk.loc[i, 'PREMIUM_CUSTOMER'] == '':
                chunk.loc[i, 'PREMIUM_CUSTOMER'] = list2[index2]
                index2 = (index2 + 1) % len(list2)
        chunk.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
        print(f"Processed and saved chunk.")
    print(f"Processing complete, data saved to: {output_file}")

def fill_number(a):
    file_path = "/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/" + a
    df = pd.read_csv(file_path)
    df['LYLTY_CARD_NBR.1'] = df['LYLTY_CARD_NBR.1'].apply(lambda x: random.randint(1000, 2373711))
    df.to_csv(file_path, index=False)
    print("Updated CSV saved.")

def customer_segment():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/transactionPurchase_behaviour_data.csv'
    merged_df = pd.read_csv(file_path)
    segment_sales = merged_df.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['TOT_SALES'].agg(['sum', 'mean']).reset_index()
    total_sales_lifestage = merged_df.groupby('LIFESTAGE')['TOT_SALES'].sum().sort_values(ascending=False)
    total_sales_premium = merged_df.groupby('PREMIUM_CUSTOMER')['TOT_SALES'].sum().sort_values(ascending=False)
    print(total_sales_lifestage)
    print(segment_sales)
    print(total_sales_premium)
    #
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x=total_sales_lifestage.index, y=total_sales_lifestage.values)
    # plt.title('Total Sales by Life Stage')
    # plt.xticks(rotation=45)
    # plt.ylabel('Total Sales')
    # plt.show()

    plt.figure(figsize=(7, 7))
    plt.pie(total_sales_premium.values, labels=total_sales_premium.index, autopct='%1.1f%%', startangle=140)
    plt.title('Total Sales by Premium Customer Status')
    plt.show()

def ProductLevel_Analysis():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/transactionPurchase_behaviour_data.csv'
    merged_df = pd.read_csv(file_path)
    merged_df['BRAND'] = merged_df['PROD_NAME'].str.split().str[0]
    if "BRAND" not in merged_df.columns:
        merged_df['BRAND'] = merged_df['BRAND']
        merged_df.to_csv(file_path, index=False)

    merged_df['PACK_SIZE'] = merged_df['PROD_NAME'].str.extract(r'(\d+g)', expand=False)
    if "PACK_SIZE" not in merged_df.columns:
        merged_df['PACK_SIZE'] = merged_df['PACK_SIZE']
        merged_df.to_csv(file_path, index=False)

    top_products = merged_df.groupby('PROD_NAME')['TOT_SALES'].sum().sort_values(ascending=False).head(10)
    print(top_products)
    brand_sales = merged_df.groupby('BRAND')['TOT_SALES'].sum().sort_values(ascending=False)
    print(brand_sales)
    packet_size_sales = merged_df.groupby('PACK_SIZE')['TOT_SALES'].sum().sort_values(ascending=False)
    print(packet_size_sales)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=packet_size_sales.index, y=packet_size_sales.values)
    plt.title('Total Sales by Packet Size')
    plt.xticks(rotation=45)
    plt.ylabel('Total Sales')
    plt.show()

#_new visualization:
def high_store_sales():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/transactionPurchase_behaviour_data.csv'
    merged_df = pd.read_csv(file_path)
    merged_df['DATE'] = pd.to_datetime(merged_df['DATE'])
    sales_by_date_store = merged_df.groupby(['DATE', 'STORE_NBR'])['TOT_SALES'].sum().reset_index()
    sales_by_date = sales_by_date_store.groupby('DATE')['TOT_SALES'].sum().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(sales_by_date['DATE'], sales_by_date['TOT_SALES'], marker='o')
    plt.title('Total Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.show()

def more_purchases():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/transactionPurchase_behaviour_data.csv'
    merged_df = pd.read_csv(file_path)
    class_purchases_by_store = merged_df.groupby(['LIFESTAGE', 'STORE_NBR'])['TOT_SALES'].sum().reset_index()
    sales_by_date = class_purchases_by_store.groupby('STORE_NBR')['TOT_SALES'].sum().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(sales_by_date['STORE_NBR'], sales_by_date['TOT_SALES'], marker='o')
    plt.title('more_purchases')
    plt.xlabel('STORE_NBR')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.show()

def mostproduct():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/transactionPurchase_behaviour_data.csv'
    merged_df = pd.read_csv(file_path)
    merged_df['brand'] = merged_df['PROD_NAME'].str.split().str[0]
    brand_sales = merged_df.groupby('brand')['TOT_SALES'].sum().reset_index()
    brand_sales = brand_sales.sort_values(by='TOT_SALES', ascending=False)
    plt.figure(figsize=(10, 8))
    plt.plot(brand_sales['brand'], brand_sales['TOT_SALES'], marker='o')
    plt.title('Total Sales by Brand')
    plt.xlabel('Brand')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def premiumlifestyle():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/transactionPurchase_behaviour_data.csv'
    merged_df = pd.read_csv(file_path)
    merged_df['brand'] = merged_df['PROD_NAME'].str.split().str[0]
    premium_lifestage_sales = merged_df.groupby(['brand', 'PREMIUM_CUSTOMER', 'LIFESTAGE'])['TOT_SALES'].sum().reset_index()
    sales_by_date = premium_lifestage_sales.groupby('PREMIUM_CUSTOMER')['TOT_SALES'].sum().reset_index()
    plt.figure(figsize=(10, 6))
    plt.bar(sales_by_date['PREMIUM_CUSTOMER'], sales_by_date['TOT_SALES'])
    plt.title('Total Sales by Premium Customer')
    plt.xlabel('PREMIUM CUSTOMER')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.show()

def premium_customer_trend():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/transactionPurchase_behaviour_data.csv'
    merged_df = pd.read_csv(file_path)
    premium_customer_trend = merged_df.groupby(['DATE', 'PREMIUM_CUSTOMER'])['TOT_SALES'].sum().unstack()
    premium_customer_trend.plot(kind='line', title='Premium Customer Purchases Over Time')
    plt.show()

def top_selling_product():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/transactionPurchase_behaviour_data.csv'
    merged_df = pd.read_csv(file_path)
    merged_df['brand'] = merged_df['PROD_NAME'].str.split().str[0]
    product_sales = merged_df.groupby('brand')['TOT_SALES'].sum().reset_index()
    product_sales = product_sales.sort_values(by='TOT_SALES', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(product_sales['brand'], product_sales['TOT_SALES'])
    plt.title('Top Selling Products by Brand')
    plt.xlabel('Brand')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def premium_customerbyproduct():
    file_path = '/Users/ehushubhamshaw/Desktop/Machine_learning/datasets/transactionPurchase_behaviour_data.csv'
    merged_df = pd.read_csv(file_path)
    merged_df['brand'] = merged_df['PROD_NAME'].str.split().str[0]
    merged_df['prem_cust'] = merged_df['PREMIUM_CUSTOMER'].sum()

    premium_customer_by_product = merged_df.groupby(['PREMIUM_CUSTOMER', 'brand'])['TOT_SALES'].sum().reset_index()
    premium_customer = premium_customer_by_product.sort_values(by='TOT_SALES', ascending=False)

    #Premium
    plt.figure(figsize=(10, 6))
    plt.bar(premium_customer['brand'], premium_customer['TOT_SALES'])
    plt.title('premium_customer by product pattern')
    plt.xlabel('bought by premium customers')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


#execution starts here below

customer_segment()
ProductLevel_Analysis()
high_store_sales()
more_purchases()
mostproduct()
premiumlifestyle()
premium_customer_trend()
top_selling_product()
premium_customerbyproduct()