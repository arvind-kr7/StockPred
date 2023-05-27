import yfinance as yf
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
# Plot the data
import matplotlib.pyplot as plt
import os

# %matplotlib inline

class Backend:
    def __init__(self, company=None):
        self.company = company or 'AMZN'

        print(f'company: {self.company}')
        self.scaler = MinMaxScaler(feature_range=(0,1))

    
    def get_data(self):
        print('downloading data')

        
        end = datetime.now()
        quote = self.company
        print(f'{end=}')
        start = datetime(end.year-2,end.month,end.day)
        print(f'{start=} , {end=}')
        
        try:
            data = yf.download(quote, start=start, end=end)
    
            if data.size == 0:
                return False
        
    
            print("data downnloaded")


            # Create a new dataframe with only the 'Close column 
            self.data = data.filter(['Close'])
            # Convert the dataframe to a numpy array
            self.dataset = self.data.values
            # Get the number of rows to train the model on
            self.training_data_len = int(np.ceil( len(self.dataset) * .95 ))

            # training_data_len

            print('data downloaded')

            return True
        

        except Exception:

            raise ValueError(f'invalid keyword {self.company}')




        


    def data_prep(self):

        print('data preparing')
        # Scale the data
        self.scaled_data = self.scaler.fit_transform(self.dataset)

        # Create the training data set 
        # Create the scaled training data set
        train_data = self.scaled_data[0:int(self.training_data_len), :]
        # Split the data into x_train and y_train data sets
        x_train = []
        y_train = []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
            if i<= 61:
                print(x_train)
                print(y_train)
                print()
                
        # Convert the x_train and y_train to numpy arrays 
        x_train, self.y_train = np.array(x_train), np.array(y_train)

        # Reshape the data
        self.x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        # x_train.shape

        print('data prepared successfully')

    def train_pred(self):

        print('training model')

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape= (self.x_train.shape[1], 1)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(32))
        model.add(Dense(1))


        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(self.x_train, self.y_train, batch_size=1, epochs=1)


        # Create the testing data set
        # Create a new array containing scaled values from index 1543 to 2002 
        test_data = self.scaled_data[self.training_data_len - 60: , :]
        # Create the data sets x_test and y_test
        x_test = []
        y_test = self.dataset[self.training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
            
        # Convert the data to a numpy array
        x_test = np.array(x_test)

        # Reshape the data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

        # Get the models predicted price values 
        predictions = model.predict(x_test)
        self.predictions = self.scaler.inverse_transform(predictions)

        # Get the root mean squared error (RMSE)
        rmse = np.sqrt(np.mean(((self.predictions - y_test) ** 2)))


        print('prediction successful')


    def save_result(self):

        print('saving result')

        train = self.data[:self.training_data_len]
        valid = self.data[self.training_data_len:]
        valid['Predictions'] = self.predictions
        # Visualize the data
        plt.figure(figsize=(16,6))
        plt.title(f'Prediction for {self.company}')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.savefig(f'static/prediction_{self.company}.png')


        print('result saved in static/prediction.png')


    def predict(self, company=None):

        if (self.company or company) is None:
            return
        
        if company is not None:
            self.company = company

        try:
            if not os.path.exists(f'static/prediction_{self.company}.png'):
                print('1. Getting Data')
                if not self.get_data():
                    return False
                print('2. Preparing Data')
                self.data_prep()
                print('3. Training Model and Predicting')
                self.train_pred()
                print('4. Saving Result')
                self.save_result()


            print("prediction sucessful for ", self.company)

            return True


        except Exception as e:

            
            print("Unexpected error occured",e)
            return False



if __name__ == "__main__":
    backend = Backend('AAPL')

    backend.predict('AMZN')












    



    