CREATE TABLE Address (
    AddressID SERIAL PRIMARY KEY,
    AddressLine1 TEXT NOT NULL, 
    AddressLine2 TEXT, 
    City TEXT NOT NULL, 
    StateProvince TEXT NOT NULL,
    CountryRegion TEXT NOT NULL,
    PostalCode TEXT NOT NULL, 
    rowguid TEXT UNIQUE NOT NULL,
    ModifiedDate TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE Customer (
    CustomerID SERIAL PRIMARY KEY,
    NameStyle INTEGER NOT NULL DEFAULT 0,
    Title TEXT, 
    FirstName TEXT NOT NULL,
    MiddleName TEXT,
    LastName TEXT NOT NULL,
    Suffix TEXT, 
    CompanyName TEXT,
    SalesPerson TEXT,
    EmailAddress TEXT, 
    Phone TEXT, 
    PasswordHash TEXT NOT NULL, 
    PasswordSalt TEXT NOT NULL,
    rowguid TEXT UNIQUE NOT NULL,
    ModifiedDate TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE CustomerAddress (
    CustomerID INTEGER NOT NULL,
    AddressID INTEGER NOT NULL,
    AddressType TEXT NOT NULL,
    rowguid TEXT UNIQUE NOT NULL,
    ModifiedDate TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (CustomerID, AddressID),
    FOREIGN KEY (CustomerID) REFERENCES Customer(CustomerID),
    FOREIGN KEY (AddressID) REFERENCES Address(AddressID)
);

CREATE TABLE ProductModel (
    ProductModelID SERIAL PRIMARY KEY,
    Name TEXT UNIQUE NOT NULL,
    CatalogDescription TEXT,
    rowguid TEXT UNIQUE NOT NULL,
    ModifiedDate TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE ProductCategory (
    ProductCategoryID SERIAL PRIMARY KEY,
    ParentProductCategoryID INTEGER,
    Name TEXT UNIQUE NOT NULL,
    rowguid TEXT UNIQUE NOT NULL,
    ModifiedDate TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ParentProductCategoryID) REFERENCES ProductCategory(ProductCategoryID)
);

CREATE TABLE ProductDescription (
    ProductDescriptionID SERIAL PRIMARY KEY,
    Description TEXT NOT NULL,
    rowguid TEXT UNIQUE NOT NULL,
    ModifiedDate TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE Product (
    ProductID SERIAL PRIMARY KEY,
    Name TEXT UNIQUE NOT NULL,
    ProductNumber TEXT UNIQUE NOT NULL, 
    Color TEXT, 
    StandardCost REAL NOT NULL,
    ListPrice REAL NOT NULL,
    Size TEXT, 
    Weight REAL,
    ProductCategoryID INTEGER,
    ProductModelID INTEGER,
    SellStartDate TIMESTAMP NOT NULL,
    SellEndDate TIMESTAMP,
    DiscontinuedDate TIMESTAMP,
    ThumbNailPhoto BYTEA,
    ThumbnailPhotoFileName TEXT,
    rowguid TEXT UNIQUE NOT NULL,
    ModifiedDate TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ProductModelID) REFERENCES ProductModel(ProductModelID),
    FOREIGN KEY (ProductCategoryID) REFERENCES ProductCategory(ProductCategoryID)
);


CREATE TABLE ProductModelProductDescription (
    ProductModelID INTEGER NOT NULL,
    ProductDescriptionID INTEGER NOT NULL,
    Culture TEXT NOT NULL, 
    rowguid TEXT UNIQUE NOT NULL,
    ModifiedDate TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ProductModelID, ProductDescriptionID, Culture),
    FOREIGN KEY (ProductDescriptionID) REFERENCES ProductDescription(ProductDescriptionID),
    FOREIGN KEY (ProductModelID) REFERENCES ProductModel(ProductModelID)
);

CREATE TABLE SalesOrderHeader (
    SalesOrderID SERIAL PRIMARY KEY,
    RevisionNumber INTEGER NOT NULL DEFAULT 0,
    OrderDate TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    DueDate TIMESTAMP NOT NULL,
    ShipDate TIMESTAMP,
    Status INTEGER NOT NULL DEFAULT 1,
    OnlineOrderFlag INTEGER NOT NULL DEFAULT 1,
    SalesOrderNumber TEXT UNIQUE NOT NULL, 
    PurchaseOrderNumber VARCHAR(20),
    AccountNumber TEXT,
    CustomerID INTEGER NOT NULL,
    ShipToAddressID INTEGER,
    BillToAddressID INTEGER,
    ShipMethod TEXT NOT NULL,
    CreditCardApprovalCode TEXT,    
    SubTotal REAL NOT NULL DEFAULT 0,
    TaxAmt REAL NOT NULL DEFAULT 0,
    Freight REAL NOT NULL DEFAULT 0,
    TotalDue REAL NOT NULL,
    Comment TEXT,
    rowguid TEXT UNIQUE NOT NULL,
    ModifiedDate TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (CustomerID) REFERENCES Customer(CustomerID),
    FOREIGN KEY (ShipToAddressID) REFERENCES Address(AddressID),
    FOREIGN KEY (BillToAddressID) REFERENCES Address(AddressID)
);

CREATE TABLE SalesOrderDetail (
    SalesOrderID INTEGER NOT NULL,
    SalesOrderDetailID SERIAL NOT NULL,
    OrderQty INTEGER NOT NULL,
    ProductID INTEGER NOT NULL,
    UnitPrice REAL NOT NULL,
    UnitPriceDiscount REAL NOT NULL DEFAULT 0,
    LineTotal REAL NOT NULL,
    rowguid TEXT UNIQUE NOT NULL,
    ModifiedDate TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (SalesOrderID, SalesOrderDetailID),
    FOREIGN KEY (SalesOrderID) REFERENCES SalesOrderHeader(SalesOrderID),
    FOREIGN KEY (ProductID) REFERENCES Product(ProductID)
);
