import pandas as pd

# Function to convert semi-structured data for new_car_detail
def convert_semi_to_structured(semi_structured_data):
    structured_data = {
        'it': semi_structured_data.get('it', None),
        'fuelType': semi_structured_data.get('ft', None),
        'bodyType': semi_structured_data.get('bt', None),
        'kilometersDriven': semi_structured_data.get('km', None),
        'transmission': semi_structured_data.get('transmission', None),
        'ownerNumber': semi_structured_data.get('ownerNo', None),
        'owner': semi_structured_data.get('owner', None),
        'oem': semi_structured_data.get('oem', None),
        'model': semi_structured_data.get('model', None),
        'modelYear': semi_structured_data.get('modelYear', None),
        'variantId': semi_structured_data.get('centralVariantId', None),
        'variantName': semi_structured_data.get('variantName', None),
        'price': semi_structured_data.get('price', None),
        'priceActual': semi_structured_data.get('priceActual', None),
        # 'priceSaving': semi_structured_data.get('priceSaving', None),
        # 'priceFixedText': semi_structured_data.get('priceFixedText', None),
        'trendingImgUrl': semi_structured_data.get('trendingText', {}).get('imgUrl', None),
        'trendingHeading': semi_structured_data.get('trendingText', {}).get('heading', None),
        'trendingDescription': semi_structured_data.get('trendingText', {}).get('desc', None)
    }
    return structured_data

# Function to extract and convert new_car_overview
def convert_overview_to_structured(overview_data):
    top_overview = {item['key']: item['value'] for item in overview_data.get('top', [])}
    structured_overview = {
        'Registration Year': top_overview.get('Registration Year', None),
        'Insurance Validity': top_overview.get('Insurance Validity', None),
        'Fuel Type': top_overview.get('Fuel Type', None),
        'Seats': top_overview.get('Seats', None),
        'Kms Driven': top_overview.get('Kms Driven', None),
        'RTO': top_overview.get('RTO', None),
        'Ownership': top_overview.get('Ownership', None),
        'Engine Displacement': top_overview.get('Engine Displacement', None),
        'Transmission': top_overview.get('Transmission', None),
        'Year of Manufacture': top_overview.get('Year of Manufacture', None)
    }
    return structured_overview

# Function to extract and convert new_car_specs
def convert_specs_to_structured(specs_data):
    top_specs = {item['key']: item['value'] for item in specs_data.get('top', [])}
    detailed_specs = {entry['subHeading']: {item['key']: item['value'] for item in entry.get('list', [])} for entry in specs_data.get('data', [])}
    
    structured_specs = {
        'Mileage': top_specs.get('Mileage', None),
        'Engine': top_specs.get('Engine', None),
        'Max Power': top_specs.get('Max Power', None),
        'Torque': top_specs.get('Torque', None),
        'Wheel Size': top_specs.get('Wheel Size', None),
        'Seats (Specs)': top_specs.get('Seats', None),
        'Engine Type': detailed_specs.get('Engine', {}).get('Engine Type', None),
        'Displacement': detailed_specs.get('Engine', {}).get('Displacement', None),
        'Max Power (Specs)': detailed_specs.get('Engine', {}).get('Max Power', None),
        'Max Torque (Specs)': detailed_specs.get('Engine', {}).get('Max Torque', None),
        'No of Cylinder': detailed_specs.get('Engine', {}).get('No of Cylinder', None),
        'Values per Cylinder': detailed_specs.get('Engine', {}).get('Values per Cylinder', None),
        'Value Configuration': detailed_specs.get('Engine', {}).get('Value Configuration', None),
        'Fuel Supply System': detailed_specs.get('Engine', {}).get('Fuel Suppy System', None),
        'Bore X Stroke': detailed_specs.get('Engine', {}).get('BoreX Stroke', None),
        'Turbo Charger': detailed_specs.get('Engine', {}).get('Turbo Charger', None),
        'Super Charger': detailed_specs.get('Engine', {}).get('Super Charger', None),
        'Length': detailed_specs.get('Dimensions', {}).get('Length', None),
        'Width': detailed_specs.get('Dimensions', {}).get('Width', None),
        'Height': detailed_specs.get('Dimensions', {}).get('Height', None),
        'Wheel Base': detailed_specs.get('Dimensions', {}).get('Wheel Base', None),
        'Front Tread': detailed_specs.get('Dimensions', {}).get('Front Tread', None),
        'Rear Tread': detailed_specs.get('Dimensions', {}).get('Rear Tread', None),
        'Kerb Weight': detailed_specs.get('Dimensions', {}).get('Kerb Weight', None),
        'Gross Weight': detailed_specs.get('Dimensions', {}).get('Gross Weight', None),
        'Gear Box': detailed_specs.get('Miscellaneous', {}).get('Gear Box', None),
        'Drive Type': detailed_specs.get('Miscellaneous', {}).get('Drive Type', None),
        'Seating Capacity': detailed_specs.get('Miscellaneous', {}).get('Seating Capacity', None),
        'Steering Type': detailed_specs.get('Miscellaneous', {}).get('Steering Type', None),
        'Turning Radius': detailed_specs.get('Miscellaneous', {}).get('Turning Radius', None),
        'Front Brake Type': detailed_specs.get('Miscellaneous', {}).get('Front Brake Type', None),
        'Rear Brake Type': detailed_specs.get('Miscellaneous', {}).get('Rear Brake Type', None),
        'Tyre Type': detailed_specs.get('Miscellaneous', {}).get('Tyre Type', None),
        'Alloy Wheel Size': detailed_specs.get('Miscellaneous', {}).get('Alloy Wheel Size', None),
        'No Door Numbers': detailed_specs.get('Miscellaneous', {}).get('No Door Numbers', None)
    }
    return structured_specs

# Function to extract and process city data
def extract_city_data(file_path, city_name):
    df = pd.read_excel(file_path)
    
    # Extract and convert semi-structured data from 'new_car_detail' column
    semi_structured_data = df['new_car_detail'].apply(eval)  # Convert string representation to dictionary
    structured_data_list = [convert_semi_to_structured(data) for data in semi_structured_data]
    structured_df = pd.DataFrame(structured_data_list)
    
    # Extract and convert semi-structured data from 'new_car_overview' column
    overview_data = df['new_car_overview'].apply(eval)  # Convert string representation to dictionary
    overview_data_list = [convert_overview_to_structured(data) for data in overview_data]
    overview_df = pd.DataFrame(overview_data_list)
    
    # Extract and convert semi-structured data from 'new_car_specs' column
    specs_data = df['new_car_specs'].apply(eval)  # Convert string representation to dictionary
    specs_data_list = [convert_specs_to_structured(data) for data in specs_data]
    specs_df = pd.DataFrame(specs_data_list)
    
    # Combine all three dataframes
    combined_df = pd.concat([structured_df, overview_df, specs_df], axis=1)
    combined_df['City'] = city_name
    
    # Ensure all expected columns are present
    expected_columns = combined_df.columns.tolist()
    for col in expected_columns:
        if col not in combined_df.columns:
            combined_df[col] = None
            
    return combined_df

# File paths and corresponding city names
files_and_cities = {
    "C:\\Users\\arund\\Downloads\\delhi_cars.xlsx": "Delhi",
    "C:\\Users\\arund\\Downloads\\hyderabad_cars.xlsx": "Hyderabad",
    "C:\\Users\\arund\\Downloads\\jaipur_cars.xlsx": "Jaipur",
    "C:\\Users\\arund\\Downloads\\kolkata_cars.xlsx": "Kolkata",
    "C:\\Users\\arund\\Downloads\\bangalore_cars.xlsx": "Bangalore",
    "C:\\Users\\arund\\Downloads\\chennai_cars.xlsx": "Chennai"
}

# Extract and process data from each file and concatenate into a single DataFrame
all_data = pd.concat([extract_city_data(file, city) for file, city in files_and_cities.items()], ignore_index=True)

# Save the merged data to a new Excel file
output_path = "C:\\Users\\arund\\Downloads\\merged_cities_data.xlsx"
all_data.to_excel(output_path, index=False)

print(f'Merged data saved to {output_path}')
# Load your dataset
file_path = "C:\\Users\\arund\\Downloads\\merged_cities_data.xlsx"
df = pd.read_excel(file_path)

# Handling missing values for numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

# Handling missing values for categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# Save the cleaned dataset
df.to_excel('C:\\Users\\arund\\Downloads\\cleaned_dataset.xlsx', index=False)

# Reload the cleaned dataset for further processing
df = pd.read_excel('C:\\Users\\arund\\Downloads\\cleaned_dataset.xlsx')

# Helper function to convert price to numeric (removing currency symbols and commas)
def convert_price(price_str):
    try:
        price_str = price_str.replace('₹', '').replace(',', '').strip()
        if 'Crore' in price_str:
            price_str = price_str.replace('Crore', '').strip()
            return pd.to_numeric(price_str, errors='coerce') * 10**7
        elif 'Lakh' in price_str:
            price_str = price_str.replace('Lakh', '').strip()
            return pd.to_numeric(price_str, errors='coerce') * 10**5
        else:
            return pd.to_numeric(price_str, errors='coerce')
    except:
        return np.nan

# Helper function to convert kilometers to numeric
def convert_kilometers(km_str):
    try:
        return pd.to_numeric(km_str.replace('Kms', '').replace('km', '').replace(',', '').strip(), errors='coerce')
    except:
        return np.nan

# Convert relevant columns to appropriate types
df['kilometersDriven'] = df['kilometersDriven'].apply(convert_kilometers)
df['price'] = df['price'].apply(convert_price)
df['priceActual'] = df['priceActual'].apply(convert_price)

# Convert other columns to numeric where applicable
numeric_columns = ['Registration Year', 'Seats', 'Kms Driven', 'Engine Displacement', 'Year of Manufacture', 
                   'Mileage', 'Max Power', 'Torque', 'Wheel Size', 'Seats (Specs)', 'Displacement', 
                   'Max Power (Specs)', 'Max Torque (Specs)', 'No of Cylinder', 'Values per Cylinder', 
                   'Length', 'Width', 'Height', 'Wheel Base', 'Front Tread', 'Rear Tread', 'Kerb Weight', 
                   'Gross Weight', 'Seating Capacity', 'Turning Radius', 'No Door Numbers']

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col].astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors='coerce')

# Convert categorical columns to 'category' type
categorical_columns = [
    'fuelType', 'bodyType', 'transmission', 'owner', 'oem', 'model', 'variantName',
    'Seats', 'RTO', 'Ownership', 'Engine', 'Engine Type', 'Fuel Supply System',
    'Turbo Charger', 'Super Charger', 'Gear Box', 'Drive Type', 'Steering Type',
    'Front Brake Type', 'Rear Brake Type', 'Tyre Type', 'Alloy Wheel Size', 'City'
]
df[categorical_columns] = df[categorical_columns].astype('category')

# One-Hot Encoding for Nominal Variables
nominal_columns = [
    'fuelType', 'bodyType', 'transmission', 'ownerNumber', 'Insurance Validity', 'oem', 'model', 'variantName',
    'RTO', 'Ownership', 'Engine', 'Engine Type', 'Fuel Supply System',
    'Turbo Charger', 'Super Charger', 'Gear Box', 'Drive Type', 'Steering Type',
    'Front Brake Type', 'Rear Brake Type', 'Tyre Type', 'Alloy Wheel Size', 'City'
]
df_nominal = pd.get_dummies(df[nominal_columns], drop_first=True)

# Label Encoding for Ordinal Variables (if any)
ordinal_columns = ['Seats']  # Adjust if there are other ordinal columns
df_ordinal = df[ordinal_columns]

# Normalizing Numerical Features
numerical_columns = [
    'kilometersDriven', 'price', 'priceActual', "ownerNumber",
    'Registration Year',  'Kms Driven', 'Engine Displacement',
    'Year of Manufacture', 'Mileage', 'Max Power', 'Torque', 'Wheel Size',
    'Seats (Specs)', 'Displacement', 'Max Power (Specs)', 'Max Torque (Specs)',
    'No of Cylinder', 'Values per Cylinder', 'Length', 'Width', 'Height',
    'Wheel Base', 'Front Tread', 'Rear Tread', 'Kerb Weight', 'Gross Weight',
    'Seating Capacity', 'Turning Radius', 'No Door Numbers'
]
scaler = MinMaxScaler()  # or StandardScaler()
df_numerical = pd.DataFrame(scaler.fit_transform(df[numerical_columns]), columns=numerical_columns)

# Combine all parts
df_final = pd.concat([df_nominal, df_ordinal, df_numerical], axis=1)

# Outlier Detection and Removal using IQR
def remove_outliers_iqr(df, numerical_columns):
    df_cleaned = df.copy()
    for col in numerical_columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        df_cleaned = df_cleaned[~((df_cleaned[col] < (Q1 - 1.5 * IQR)) | (df_cleaned[col] > (Q3 + 1.5 * IQR)))]
    return df_cleaned

# Remove outliers
df_cleaned = remove_outliers_iqr(df, numerical_columns)

# Reapply encoding and scaling after removing outliers
df_nominal_cleaned = pd.get_dummies(df_cleaned[nominal_columns], drop_first=True)
df_ordinal_cleaned = df_cleaned[ordinal_columns]
df_numerical_cleaned = pd.DataFrame(scaler.fit_transform(df_cleaned[numerical_columns]), columns=numerical_columns)

# Combine all parts
df_final = pd.concat([df_nominal_cleaned, df_ordinal_cleaned, df_numerical_cleaned], axis=1)
