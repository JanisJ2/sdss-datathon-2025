# 🏡 Toronto Real Estate Price Prediction with XGBoost
**Students in Data Science and Statistics | University of Toronto**

---

## 📌 Overview
This project develops an **XGBoost-based real estate price prediction model** for **Toronto**, leveraging **geospatial insights**. Instead of using only **ward information** or raw **latitude/longitude**, we enhance the dataset by mapping each listing to its **Toronto Open Data Neighbourhood**. This provides **richer location-based features** that improve model accuracy.

---

## 📂 Dataset
### **1️⃣ Real Estate Data (`real-estate-data.csv`)**
A **simulated dataset** representing Toronto’s real estate market across multiple neighbourhoods.  
📌 **Key features**:
- 🏡 **Property details**: `beds`, `baths`, `size`, `parking`, `exposure`, `DEN`
- 📈 **Market data**: `D_mkt` (days on market), `maint` (maintenance fee)
- 🌍 **Location data**: `ward`, `latitude (lt)`, `longitude (lg)`
- 🏙 **Enhanced location**: **Neighbourhood assigned from Toronto Open Data**
- 💰 **Target variable**: `price` (listing price)

📖 *For column details, see* **[Real Estate Data Dictionary](Real%20Estate%20Data%20Dictionary.pdf)**.

### **2️⃣ Toronto Open Data Neighbourhoods**
- 🏙 **Neighbourhood Mapping** → Listings categorized into **official Toronto neighbourhoods**.
- 📊 **Rich Location Context** → Each listing now belongs to a **neighbourhood** rather than just a ward.
- 🌍 **Better Geospatial Representation** → More granular than latitude/longitude alone.

---

## 🛠 Methodology
### **1️⃣ Data Preprocessing & Feature Engineering**
🔧 **Handled missing values, categorical encoding**  
🔄 **Mapped each listing to its corresponding neighbourhood**  

---

### **2️⃣ Exploratory Data Analysis (EDA)**
📊 **Key Insights from EDA**:
- **Price Trends Across Neighbourhoods** 🏡 → Some areas have consistently higher values.
- **Impact of Property Size** 📍 → Larger properties show higher median prices.

📚 *See full analysis in the Jupyter Notebook:*  
📂 **[SDSS Datathon.ipynb](SDSS%20Datathon.ipynb)**

---

### **3️⃣ Predictive Modeling (XGBoost)**
#### **Why XGBoost?**
🔥 **Handles large, structured datasets efficiently**  
🔥 **Boosting prevents overfitting**  
🔥 **Works well with categorical & numerical data**  

#### **Model Training:**
- **Feature Selection** 📊 → Based on correlation heatmaps  
- **Hyperparameter Tuning** 🔧 → GridSearchCV  
- **Validation Metrics** 📈 → RMSE, R², MAE  

💪 **Final Model Performance**  
📉 **RMSE**: _X_  
📊 **R² Score**: _X_  

---

## 💚 Running the Model
### **1️⃣ Install Dependencies**
```bash
pip install pandas numpy scikit-learn xgboost seaborn pydeck
```
### **2️⃣ Run the Prediction Model**
```bash
python [to be filled later].py
```
### **3️⃣ View Interactive Geospatial Map**
```bash
open geospatial.html
```

---

## 👨‍💻 Contributors
- **shrimp_net2.0**
- **Christoffer Tan, Faraaz Ahmed, Janis Joplin, Nagata Aptana**
- **University of Toronto, Data Science & Statistics**
- **March 2025**

---

## 💎 References
- **Toronto Open Data**: [https://open.toronto.ca](https://open.toronto.ca)
