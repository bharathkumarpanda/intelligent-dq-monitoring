from hdbcli import dbapi
import pandas as pd

# Connect to SAP HANA Cloud
conn = dbapi.connect(
    address="82ad1b10-39a6-462a-bf26-a0eb6609131b.hana.prod-ap21.hanacloud.ondemand.com",
    port=443,
    user="DBADMIN",
    password="Dqmonitoring1!",
    encrypt=True,
    sslValidateCertificate=False
)
print("Connected to SAP HANA Cloud!")

cursor = conn.cursor()

# Drop old table if exists
try:
    cursor.execute("DROP TABLE SUPPLY_CHAIN_DATA")
    print("Old table dropped.")
except:
    print("No old table found.")

# Step 1 - Load CSV to check columns
df = pd.read_csv("supply_chain_data.csv")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns from CSV")

# Step 2 - Create table with all NVARCHAR columns (safe for any data)
columns = []
for col in df.columns:
    safe_name = col.strip().replace(" ", "_").replace("-", "_").upper()
    columns.append(f'"{safe_name}" NVARCHAR(500)')

create_sql = f'CREATE TABLE SUPPLY_CHAIN_DATA ({", ".join(columns)})'
cursor.execute(create_sql)
print("Table created!")

# Step 3 - Insert data
placeholders = ",".join(["?" for _ in df.columns])
count = 0
for index, row in df.iterrows():
    values = [str(val) if pd.notna(val) else None for val in row]
    cursor.execute(f"INSERT INTO SUPPLY_CHAIN_DATA VALUES ({placeholders})", values)
    count += 1

conn.commit()
print(f"Inserted {count} rows into SAP HANA Cloud!")

# Step 4 - Verify
cursor.execute("SELECT COUNT(*) FROM SUPPLY_CHAIN_DATA")
result = cursor.fetchone()
print(f"Total rows in HANA table: {result[0]}")

cursor.close()
conn.close()
print("Done! Your data is now in SAP HANA Cloud!") 