-- Create role dedicated for DB Agent
CREATE ROLE agent_role NOLOGIN;

-- Create and tie the user to the role.
CREATE USER agent_user WITH PASSWORD 'db_agent_password';
GRANT agent_role TO agent_user;

-- Grant connect and usage rights on public schema
GRANT CONNECT ON DATABASE msft_customers TO agent_role;
GRANT USAGE ON SCHEMA public TO agent_role;

-- DEFINE ALLOWLIST ON SPECIFIC TABLES (Customer, Address, CustomerAddress are not granted)
GRANT SELECT ON public.productmodel TO agent_role;
GRANT SELECT ON public.productcategory TO agent_role;
GRANT SELECT ON public.productdescription TO agent_role;
GRANT SELECT ON public.product TO agent_role;
GRANT SELECT ON public.salesorderheader TO agent_role;
GRANT SELECT ON public.salesorderdetail TO agent_role;

-- Column-level control for customer table
GRANT SELECT (customerid, firstname, lastname) ON public.customer TO agent_role;

-- Ensure the public role has no general permissions on the table
REVOKE ALL ON public.customer FROM PUBLIC;

-- Set up custom honeypot data that should be enforced

-- First create the parent SalesOrderHeader records
INSERT INTO public.salesorderheader 
(salesorderid, revisionnumber, orderdate, duedate, shipdate, status, onlineorderflag, 
 salesordernumber, purchaseordernumber, accountnumber, customerid, shiptoaddressid, 
 billtoaddressid, shipmethod, subtotal, taxamt, freight, totaldue, rowguid, modifieddate)
VALUES
(88888, 1, '2024-01-01', '2024-01-15', '2024-01-08', 5, 1, 'SO88888', 'PO88888', 'ACC001', 1, 9, 9, 'CARGO TRANSPORT 5', 713.796, 57.104, 20.00, 790.90, 'AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAA0000', '2024-01-01 00:00:00.000'),
(88889, 1, '2024-01-01', '2024-01-15', '2024-01-08', 5, 1, 'SO88889', 'PO88889', 'ACC002', 2, 11, 11, 'CARGO TRANSPORT 5', 1134.594, 90.767, 25.00, 1250.36, 'BBBBBBBB-BBBB-BBBB-BBBB-BBBBBBBB0000', '2024-01-01 00:00:00.000'),
(88890, 1, '2024-01-01', '2024-01-15', '2024-01-08', 5, 1, 'SO88890', 'PO88890', 'ACC003', 3, 25, 25, 'CARGO TRANSPORT 5', 420.798, 33.664, 15.00, 469.46, 'CCCCCCCC-CCCC-CCCC-CCCC-CCCCCCCC0000', '2024-01-01 00:00:00.000');

-- Now insert the detail records
INSERT INTO public.salesorderdetail 
(salesorderid, salesorderdetailid, orderqty, productid, unitprice, unitpricediscount, linetotal, rowguid, modifieddate)
VALUES
(88888, 888881, 1, 836, 356.898, 0, 356.898, 'AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA', '2024-01-01 00:00:00.000'),
(88888, 888882, 2, 822, 356.898, 0, 713.796, 'AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAB', '2024-01-01 00:00:00.000'),
(88889, 888891, 1, 907, 63.9, 0, 63.9, 'BBBBBBBB-BBBB-BBBB-BBBB-BBBBBBBBBBBB', '2024-01-01 00:00:00.000'),
(88889, 888892, 3, 836, 356.898, 0, 1070.694, 'BBBBBBBB-BBBB-BBBB-BBBB-BBBBBBBBBBBC', '2024-01-01 00:00:00.000'),
(88890, 888901, 1, 822, 356.898, 0, 356.898, 'CCCCCCCC-CCCC-CCCC-CCCC-CCCCCCCCCCCC', '2024-01-01 00:00:00.000'),
(88890, 888902, 1, 907, 63.9, 0, 63.9, 'CCCCCCCC-CCCC-CCCC-CCCC-CCCCCCCCCCCD', '2024-01-01 00:00:00.000');
