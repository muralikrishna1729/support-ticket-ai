import psycopg2

conn = psycopg2.connect(
    host="smartticket-db.cx4mkqkgqsxv.ap-south-1.rds.amazonaws.com",
    port=5432,
    dbname="smartticket",
    user="ticketuser",
    password="ticketpass123"
)
print("Connected successfully!")
conn.close()