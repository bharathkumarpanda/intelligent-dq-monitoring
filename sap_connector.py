from hdbcli import dbapi

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
cursor.execute("SELECT CURRENT_USER, CURRENT_TIMESTAMP FROM DUMMY")
row = cursor.fetchone()
print(f"User: {row[0]}")
print(f"Time: {row[1]}")

cursor.close()
conn.close()
print("Connection closed.") 