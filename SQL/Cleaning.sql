-- FINAL DATA CLEANING SCRIPT - SQL SERVER
-- DATA_PROJECT: AI4I 2020 Predictive Maintenance
-- DATABASE: Machine_Failure_Project

-- Create Database

CREATE DATABASE Machine_Failure_Project;
GO

USE Machine_Failure_Project;
GO

-- Create Table

CREATE TABLE predictive_maintenance (

    UDI INT,
    Product_ID VARCHAR(20),
    Type VARCHAR(2),

    Air_Temperature FLOAT,
    Process_Temperature FLOAT,
    Rotational_Speed FLOAT,
    Torque FLOAT,
    Tool_Wear FLOAT,

    Machine_Failure INT,

    TWF INT,
    HDF INT,
    PWF INT,
    OSF INT,
    RNF INT

);
GO

-- Import CSV

BULK INSERT predictive_maintenance
FROM "C:\Users\Ayakhaled\Downloads\DPI_Course\Manufacturing_Project\data\raw\ai4i2020.csv"
WITH
(
FIELDTERMINATOR=',',
ROWTERMINATOR='\n',
FIRSTROW=2
);
GO

-- Preview Dataset

SELECT TOP 5 *
FROM predictive_maintenance;
GO

-- Dataset Information

SELECT COUNT(*) AS Total_Rows
FROM predictive_maintenance;
GO

-- Check Missing Values

SELECT

COUNT(*) AS Total_Rows,

SUM(CASE WHEN UDI IS NULL THEN 1 ELSE 0 END) Missing_UDI,

SUM(CASE WHEN Product_ID IS NULL THEN 1 ELSE 0 END) Missing_Product_ID,

SUM(CASE WHEN Type IS NULL THEN 1 ELSE 0 END) Missing_Type,

SUM(CASE WHEN Air_Temperature IS NULL THEN 1 ELSE 0 END) Missing_AirTemp,

SUM(CASE WHEN Process_Temperature IS NULL THEN 1 ELSE 0 END) Missing_ProcessTemp,

SUM(CASE WHEN Rotational_Speed IS NULL THEN 1 ELSE 0 END) Missing_RPM,

SUM(CASE WHEN Torque IS NULL THEN 1 ELSE 0 END) Missing_Torque,

SUM(CASE WHEN Tool_Wear IS NULL THEN 1 ELSE 0 END) Missing_ToolWear,

SUM(CASE WHEN Machine_Failure IS NULL THEN 1 ELSE 0 END) Missing_Target

FROM predictive_maintenance;
GO

-- Check Duplicate Records

SELECT

COUNT(*) AS Total_Rows,

COUNT(DISTINCT

CAST(UDI AS VARCHAR)+'|'+
Product_ID+'|'+
Type+'|'+

CAST(Air_Temperature AS VARCHAR)+'|'+
CAST(Process_Temperature AS VARCHAR)+'|'+
CAST(Rotational_Speed AS VARCHAR)+'|'+
CAST(Torque AS VARCHAR)+'|'+
CAST(Tool_Wear AS VARCHAR)+'|'+

CAST(Machine_Failure AS VARCHAR)

) AS Unique_Rows

FROM predictive_maintenance;
GO


-- Calculate Quartiles (Q1 & Q3)

WITH Quartiles AS
(

SELECT DISTINCT

PERCENTILE_CONT(0.25) WITHIN GROUP
(ORDER BY Air_Temperature) OVER() Air_Q1,

PERCENTILE_CONT(0.75) WITHIN GROUP
(ORDER BY Air_Temperature) OVER() Air_Q3,

PERCENTILE_CONT(0.25) WITHIN GROUP
(ORDER BY Process_Temperature) OVER() Process_Q1,

PERCENTILE_CONT(0.75) WITHIN GROUP
(ORDER BY Process_Temperature) OVER() Process_Q3,

PERCENTILE_CONT(0.25) WITHIN GROUP
(ORDER BY Rotational_Speed) OVER() RPM_Q1,

PERCENTILE_CONT(0.75) WITHIN GROUP
(ORDER BY Rotational_Speed) OVER() RPM_Q3,

PERCENTILE_CONT(0.25) WITHIN GROUP
(ORDER BY Torque) OVER() Torque_Q1,

PERCENTILE_CONT(0.75) WITHIN GROUP
(ORDER BY Torque) OVER() Torque_Q3,

PERCENTILE_CONT(0.25) WITHIN GROUP
(ORDER BY Tool_Wear) OVER() Wear_Q1,

PERCENTILE_CONT(0.75) WITHIN GROUP
(ORDER BY Tool_Wear) OVER() Wear_Q3

FROM predictive_maintenance

)

SELECT *

FROM Quartiles;
GO


-- Calculate IQR and Bounds

WITH Quartiles AS
(

SELECT DISTINCT

PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Air_Temperature) OVER() AQ1,
PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Air_Temperature) OVER() AQ3,

PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Process_Temperature) OVER() PQ1,
PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Process_Temperature) OVER() PQ3,

PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Rotational_Speed) OVER() RQ1,
PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Rotational_Speed) OVER() RQ3,

PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Torque) OVER() TQ1,
PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Torque) OVER() TQ3,

PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Tool_Wear) OVER() WQ1,
PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Tool_Wear) OVER() WQ3

FROM predictive_maintenance

)

SELECT

AQ1,AQ3,
AQ3-AQ1 Air_IQR,
AQ1-1.5*(AQ3-AQ1) Air_Lower,
AQ3+1.5*(AQ3-AQ1) Air_Upper,

PQ1,PQ3,
PQ3-PQ1 Process_IQR,
PQ1-1.5*(PQ3-PQ1) Process_Lower,
PQ3+1.5*(PQ3-PQ1) Process_Upper,

RQ1,RQ3,
RQ3-RQ1 RPM_IQR,
RQ1-1.5*(RQ3-RQ1) RPM_Lower,
RQ3+1.5*(RQ3-RQ1) RPM_Upper,

TQ1,TQ3,
TQ3-TQ1 Torque_IQR,
TQ1-1.5*(TQ3-TQ1) Torque_Lower,
TQ3+1.5*(TQ3-TQ1) Torque_Upper,

WQ1,WQ3,
WQ3-WQ1 Wear_IQR,
WQ1-1.5*(WQ3-WQ1) Wear_Lower,
WQ3+1.5*(WQ3-WQ1) Wear_Upper

FROM Quartiles;
GO

-- Count Outliers

WITH Limits AS
(

SELECT DISTINCT

PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Air_Temperature) OVER() AQ1,
PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Air_Temperature) OVER() AQ3,

PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Process_Temperature) OVER() PQ1,
PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Process_Temperature) OVER() PQ3,

PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Rotational_Speed) OVER() RQ1,
PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Rotational_Speed) OVER() RQ3,

PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Torque) OVER() TQ1,
PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Torque) OVER() TQ3,

PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Tool_Wear) OVER() WQ1,
PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Tool_Wear) OVER() WQ3

FROM predictive_maintenance

)

SELECT

SUM(CASE WHEN Air_Temperature < AQ1-1.5*(AQ3-AQ1)
OR Air_Temperature > AQ3+1.5*(AQ3-AQ1)
THEN 1 ELSE 0 END) AS Air_Outliers,

SUM(CASE WHEN Process_Temperature < PQ1-1.5*(PQ3-PQ1)
OR Process_Temperature > PQ3+1.5*(PQ3-PQ1)
THEN 1 ELSE 0 END) AS Process_Outliers,

SUM(CASE WHEN Rotational_Speed < RQ1-1.5*(RQ3-RQ1)
OR Rotational_Speed > RQ3+1.5*(RQ3-RQ1)
THEN 1 ELSE 0 END) AS RPM_Outliers,

SUM(CASE WHEN Torque < TQ1-1.5*(TQ3-TQ1)
OR Torque > TQ3+1.5*(TQ3-TQ1)
THEN 1 ELSE 0 END) AS Torque_Outliers,

SUM(CASE WHEN Tool_Wear < WQ1-1.5*(WQ3-WQ1)
OR Tool_Wear > WQ3+1.5*(WQ3-WQ1)
THEN 1 ELSE 0 END) AS ToolWear_Outliers

FROM predictive_maintenance
CROSS JOIN Limits;
GO

-- Display All Outlier Records

WITH Limits AS
(

SELECT DISTINCT

PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Air_Temperature) OVER() AQ1,
PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Air_Temperature) OVER() AQ3,

PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Process_Temperature) OVER() PQ1,
PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Process_Temperature) OVER() PQ3,

PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Rotational_Speed) OVER() RQ1,
PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Rotational_Speed) OVER() RQ3,

PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Torque) OVER() TQ1,
PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Torque) OVER() TQ3,

PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Tool_Wear) OVER() WQ1,
PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Tool_Wear) OVER() WQ3

FROM predictive_maintenance

)

SELECT *

FROM predictive_maintenance
CROSS JOIN Limits

WHERE

Air_Temperature < AQ1-1.5*(AQ3-AQ1)
OR Air_Temperature > AQ3+1.5*(AQ3-AQ1)

OR Process_Temperature < PQ1-1.5*(PQ3-PQ1)
OR Process_Temperature > PQ3+1.5*(PQ3-PQ1)

OR Rotational_Speed < RQ1-1.5*(RQ3-RQ1)
OR Rotational_Speed > RQ3+1.5*(RQ3-RQ1)

OR Torque < TQ1-1.5*(TQ3-TQ1)
OR Torque > TQ3+1.5*(TQ3-TQ1)

OR Tool_Wear < WQ1-1.5*(WQ3-WQ1)
OR Tool_Wear > WQ3+1.5*(WQ3-WQ1);
GO




-- Remove unnecessary columns

ALTER TABLE predictive_maintenance
DROP COLUMN UDI, Product_ID;
GO



-- Rename failure mode columns

EXEC sp_rename
'predictive_maintenance.TWF',
'Tool_Wear_Failure',
'COLUMN';

EXEC sp_rename
'predictive_maintenance.HDF',
'Heat_Dissipation_Failure',
'COLUMN';

EXEC sp_rename
'predictive_maintenance.PWF',
'Power_Failure',
'COLUMN';

EXEC sp_rename
'predictive_maintenance.OSF',
'Overstrain_Failure',
'COLUMN';

EXEC sp_rename
'predictive_maintenance.RNF',
'Random_Failure',
'COLUMN';
GO


-- Final Preview

SELECT *
FROM predictive_maintenance;
GO










