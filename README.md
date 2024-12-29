# Artificial-Intelligence-Based-Investment-Prediction
An AI-based financial forecasting project that predicts future investment values for USD, EUR, and gold. Using LSTM models, the project provides insights into potential returns based on historical data, enabling users to simulate and visualize their investment growth over time."
## Prerequisites

Before running the project, make sure you have the following installed on your system:

- **Python 3.x**
- **pip** (Python package installer)

You also need to install the required libraries, which you can do using the `requirements.txt` file included in the repository.

## Project Files

This project requires the following files:

- **duzenlenmis_veri_dolar1.csv**: Historical data for Dolar.
- **duzenlenmis_veri_euro1.csv**: Historical data for Euro.
- **duzenlenmis_veri_altın1.csv**: Historical data for Altın.
- **investment_prediction.py**: The Python script to run the machine learning model.

Make sure you have these files in the project folder before proceeding.

---

## How to Run the Project

### 1. **Clone the Repository**
Clone the repository to your local machine by using the following command:

```bash
git clone https://github.com/tulinsymn/Artificial-Intelligence-Based-Investment-Prediction.git
cd Artificial-Intelligence-Based-Investment-Prediction
2. Install Dependencies
Make sure to install the required libraries for the project. You can use the requirements.txt file to install them:

bash
Kodu kopyala
pip install -r requirements.txt
The required libraries are:

pandas
numpy
matplotlib
scikit-learn
tensorflow
These libraries can also be installed individually by using the following commands:

bash
Kodu kopyala
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install tensorflow
3. Download the Data Files
You will need the following CSV files for the project to work:

duzenlenmis_veri_dolar1.csv
duzenlenmis_veri_euro1.csv
duzenlenmis_veri_altın1.csv
These files contain the historical data for Dolar, Euro, and Altın. Download them from the repository or another trusted source, and place them in the same directory as the Python script.

4. Run the Python Script
Once the dependencies are installed and the necessary files are downloaded, run the Python script to start the investment prediction:

bash
Kodu kopyala
python investment_prediction.py
5. Provide Input
The script will prompt you to select an investment asset (Dolar, Euro, or Altın), enter the starting year, and the amount of capital you want to invest.

For example:

text
Kodu kopyala
Select investment type (dolar/euro/altın): dolar
Enter the starting year: 2015
Enter the investment capital (in TL): 1000
After providing this information, the script will process the data, train the machine learning model, and make predictions for the future years.
