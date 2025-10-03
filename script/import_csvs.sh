#!/bin/bash
set -e

# Use env vars or defaults
PGUSER=${PGUSER:-julia}
PGDATABASE=${PGDATABASE:-msft_db}

echo "Using user: $PGUSER"
echo "Using database: $PGDATABASE"

echo "Starting import of CSV files..."

import_csv() {
  local table=$1
  local file=$2
  echo "Importing $file into table $table..."
  psql -U "$PGUSER" -d "$PGDATABASE" -c "\COPY $table FROM '$file' CSV HEADER NULL '';"
}

# Import tables in dependency order
import_csv ProductModel /app/data/ProductModel.csv
import_csv ProductCategory /app/data/ProductCategory.csv
import_csv ProductDescription /app/data/ProductDescription.csv
import_csv Product /app/data/Product.csv
import_csv Customer /app/data/Customer.csv
import_csv Address /app/data/Address.csv
import_csv CustomerAddress /app/data/CustomerAddress.csv
import_csv SalesOrderHeader /app/data/SalesOrderHeader.csv
import_csv SalesOrderDetail /app/data/SalesOrderDetail.csv

echo "CSV import complete."
