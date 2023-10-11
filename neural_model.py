import asyncio
import time
import seaborn
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import scipy.stats as sts
import matplotlib.pyplot as plt
from kucoin.client import Client
from pytrends.request import TrendReq
from kucoin.asyncio import KucoinSocketManager
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import scatter_matrix
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

api_key = '626406a4e78eee0001903ed1'
api_secret = '8501ef70-9faf-47a5-8365-e26e5a1af4ec'
api_passphrase = 'Bat20020920'
client = Client(api_key, api_secret, api_passphrase)

async def main():
    global loop
    # callback function that receives messages from the socket
    async def handle_evt(msg):
        if msg['topic'] == '/market/candles:BTC-USDT_1hour':
            print(f'got candles_BTC-USDT_1hour:{msg["data"]}')

        # if msg['topic'] == '/spotMarket/level2Depth50:XTM-USDT':
        #     print(f'got Depth50:BTC-USDT:{msg["data"]}')

    ksm = await KucoinSocketManager.create(loop, client, handle_evt)
    await ksm.subscribe('/market/candles:BTC-USDT_1hour')
    await ksm.subscribe('/spotMarket/level2Depth50:XTM-USDT')

    # while True:
    #     print("sleeping to keep loop open")
    #     await asyncio.sleep(20, loop=loop)


# if __name__ == "__main__":
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(main())

def google_trends_range_info(time_from: time.gmtime(), time_to: time.gmtime()):
    pytrend = TrendReq()
    key_words = ['bitcoin']
    data = pytrend.get_historical_interest(keywords=key_words, year_start=time_from.tm_year,
                                           month_start=time_from.tm_mon, day_start=time_from.tm_mday,
                                           year_end=time_to.tm_year, month_end=time_to.tm_mon,
                                           day_end=time_to.tm_mday, frequency='hourly')
    data.to_excel('C:/Users/User/Desktop/BTC_Trends.xlsx')
    return data

def get_kline_data(symbol:str,kline_type:str,size:int):
    time_now = time.time()
    data=[]
    for i in range(size):
        end=int(time_now-i*86400*62)
        start=int(end-86400*62)
        for item in client.get_kline_data(symbol=symbol,kline_type=kline_type,start=start,end=end):
            data.append(item)
    data=np.array(data[::-1])
    data_frame=pd.DataFrame(data=data,columns=['open_secs','open_price','closing_price','highest_prise','lowest_price','coin_amount','volume'])
    return data_frame

def update_history_data(data_frame:pd.DataFrame):
    sec_start=data_frame.loc[len(data_frame)-1][0]
    time_start=data_frame.loc[len(data_frame)-1][-2]
    last_google_trend=data_frame.loc[len(data_frame)-1][-1]
    kline_data=client.get_kline_data(symbol='BTC-USDT', kline_type='1hour', start=int(sec_start), end=int(time.time())-3600)[::-1][1:]
    google_trend_data=google_trends_range_info(time.gmtime(sec_start),time.gmtime(int(time.time())+86400))
    data=google_trend_data[google_trend_data.index >time_start][ google_trend_data['isPartial']==False]
    rates=np.array(data['bitcoin'])*last_google_trend / google_trend_data[google_trend_data.index ==time_start]['bitcoin'][0]
    for i in range(len(rates)):
        kline_data[i].append(data.index[i])
        kline_data[i].append(rates[i])
        data_frame.loc[len(data_frame)] = kline_data[i]
    data_frame.to_excel('C:/Users/User/Desktop/BTC_USDT_last_update.xlsx')
    return data_frame


def creation_fechs(data_frame):
    columns=list(data_frame.columns)
    fechs=['highlow','body','del_vol','rel_vol','rel_high_low','rel_open_close',
           'rel_ggl_rate','del_ggl_rate','rel_highlow_open','rel_highlow_close','rel_body_open',
           'rel_body_close','del_coin_amount','rel_coin_amount','msa7h_prise','msa7h_volume','msa7h_ggl_rate',
           'per_rel_vol','per_rel_high_low','per_rel_open_close','per_rel_ggl_rate','per_rel_highlow_open',
           'per_rel_highlow_close','per_rel_body_open','per_rel_body_close','per_rel_coin_amount']
    for i in fechs:columns.append(i)
    
    open_price=list(data_frame.open_price.astype(float))
    closing_price=list(data_frame.closing_price.astype(float))
    highest_prise=list(data_frame.highest_prise.astype(float))
    lowest_price=list(data_frame.lowest_price.astype(float))
    volume=list(data_frame.volume.astype(float))
    google_rate=list(data_frame.google_rate.astype(float))
    coin_amount=list(data_frame.coin_amount.astype(float))
    
    len_data=len(data_frame)
    
    highlow=[]
    for i,j in zip(highest_prise,lowest_price):highlow.append(i-j)
    
    body=[]
    for i,j in zip(open_price,closing_price):body.append(i-j)
        
    del_vol=[None]
    for i in range(1,len_data):del_vol.append(volume[i]-volume[i-1])
        
    rel_vol=[None]
    for i in range(1,len_data):rel_vol.append(volume[i]/volume[i-1])
        
    rel_high_low=[]
    for i,j in zip(highest_prise,lowest_price):rel_high_low.append(i/j)
        
    rel_open_close=[]
    for i,j in zip(open_price,closing_price):rel_open_close.append(i/j)
        
    rel_ggl_rate=[None]
    for i in range(1,len_data):rel_ggl_rate.append(google_rate[i]/google_rate[i-1])
        
    del_ggl_rate=[None]
    for i in range(1,len_data):del_ggl_rate.append(google_rate[i]-google_rate[i-1])
        
    rel_highlow_open=[None]
    for i in range(1,len_data):rel_highlow_open.append(highlow[i]/open_price[i])
        
    rel_highlow_close=[None]
    for i in range(1,len_data):rel_highlow_close.append(highlow[i]/closing_price[i])
        
    rel_body_open=[]
    for i,j in zip(body,open_price):rel_body_open.append(i/j)
        
    rel_body_close=[]
    for i,j in zip(body,closing_price):rel_body_close.append(i/j)
        
    del_coin_amount=[None]
    for i in range(1,len_data):del_coin_amount.append(coin_amount[i]-coin_amount[i-1])
        
    rel_coin_amount=[None]
    for i in range(1,len_data):rel_coin_amount.append(coin_amount[i]/coin_amount[i-1])
    
        
    msa7h_prise=[None]*7
    for i in range(len_data-7):
        msa=0
        for j in range(7):
            msa+=closing_price[i+j]
        msa7h_prise.append(msa/7)
    
    msa7h_volume=[None]*7
    for i in range(len_data-7):
        msa=0
        for j in range(7):
            msa+=volume[i+j]
        msa7h_volume.append(msa/7)
    
    msa7h_ggl_rate=[None]*7
    for i in range(len_data-7):
        msa=0
        for j in range(7):
            msa+=google_rate[i+j]
        msa7h_ggl_rate.append(msa/7)
        
    
    per_rel_vol=[None]
    for i in range(1,len_data):per_rel_vol.append(volume[i]/volume[i-1]*100-100)
        
    per_rel_high_low=[]
    for i,j in zip(highest_prise,lowest_price):per_rel_high_low.append(i/j*100-100)
        
    per_rel_open_close=[]
    for i,j in zip(open_price,closing_price):per_rel_open_close.append(i/j*100-100)
        
    per_rel_ggl_rate=[None]
    for i in range(1,len_data):per_rel_ggl_rate.append(google_rate[i]/google_rate[i-1]*100-100)
    
    per_rel_highlow_open=[None]
    for i in range(1,len_data):per_rel_highlow_open.append(highlow[i]/open_price[i]*100)
        
    per_rel_highlow_close=[None]
    for i in range(1,len_data):per_rel_highlow_close.append(highlow[i]/closing_price[i]*100)
        
    per_rel_body_open=[]
    for i,j in zip(body,open_price):per_rel_body_open.append(i/j*100)
        
    per_rel_body_close=[]
    for i,j in zip(body,closing_price):per_rel_body_close.append(i/j*100)
        
    per_rel_coin_amount=[None]
    for i in range(1,len_data):per_rel_coin_amount.append(coin_amount[i]/coin_amount[i-1]*100-100)
        
        
        
    full_frame=pd.DataFrame(columns=columns)
    full_frame.open_secs=data_frame.open_secs
    full_frame.open_time=data_frame.open_time
    
    full_frame.open_price=open_price
    full_frame.closing_price=closing_price
    full_frame.highest_prise=highest_prise
    full_frame.lowest_price=lowest_price
    full_frame.coin_amount=coin_amount
    full_frame.volume=volume
    full_frame.google_rate=google_rate
    
    full_frame.highlow=highlow
    full_frame.body=body
    full_frame.del_vol=del_vol
    full_frame.rel_vol=rel_vol
    full_frame.rel_high_low=rel_high_low
    full_frame.rel_open_close=rel_open_close
    full_frame.rel_ggl_rate=rel_ggl_rate
    full_frame.del_ggl_rate=del_ggl_rate
    full_frame.rel_highlow_open=rel_highlow_open
    full_frame.rel_highlow_close=rel_highlow_close
    full_frame.rel_body_open=rel_body_open
    full_frame.rel_body_close=rel_body_close
    full_frame.del_coin_amount=del_coin_amount
    full_frame.rel_coin_amount=rel_coin_amount
    
    full_frame.msa7h_prise=msa7h_prise
    full_frame.msa7h_volume=msa7h_volume
    full_frame.msa7h_ggl_rate=msa7h_ggl_rate
    
    full_frame.per_rel_vol=per_rel_vol
    full_frame.per_rel_high_low=per_rel_high_low
    full_frame.per_rel_open_close=per_rel_open_close
    full_frame.per_rel_ggl_rate=per_rel_ggl_rate
    full_frame.per_rel_highlow_open=per_rel_highlow_open
    full_frame.per_rel_highlow_close=per_rel_highlow_close
    full_frame.per_rel_body_open=per_rel_body_open
    full_frame.per_rel_body_close=per_rel_body_close
    full_frame.per_rel_coin_amount=per_rel_coin_amount
    
    return full_frame


def create_wind_data_to_RNN(data,labels,wind_size):
    for i in range(len(labels)-wind_size):
        X=[]
        y=[]
        X.append(data[i:i+wind_size])
        y.append(labels[i+wind_size])
    return tf.Tensor(X),tf.Tensor(y)

def create_wind_data(data,labels,wind_size):
    for i in range(len(answers)-wind_size):
        array=[]
        for item in data[i:i+wind_size]:
            for el in item:array.append(el)
        X.append(array)
        y.append(labels[i+wind_size])
    return tf.Tensor(X),tf.Tensor(y)

###########################################################################################################################################################################################################################

BLOCK_SIZE=120
WIND_SIZE=16
EPOCHS=20
BATCH_SIZE=16
print(f'COFIG :: block {BLOCK_SIZE}, wind {WIND_SIZE}, epoches {EPOCHS}, batch {BATCH_SIZE}')

MODEL= tf.keras.models.Sequential([
        tf.keras.layers.LSTM(20,return_sequences=True),
        tf.keras.layers.LSTM(10),
        tf.keras.layers.Dense(10,activation='ReLU'),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])        


train_accuracy=[]
test_accuracy=[]
preds=[]
y_rights=[]

data=full_frame.loc[6500:]
data.index=range(len(data))
labels=[]
for i in data.body:
    if i>0:labels.append(1)
    else:labels.append(0)
#'highlow,'del_coin_amount'
print('start fitting...')
for i in range(len(data)-BLOCK_SIZE):
    if(i>9):break

    print('BLOCK #',i+1)

    X_data=data[['body','del_vol','highlow']].loc[i:i+BLOCK_SIZE-1]
    size=len(data)-2
    scaler=MinMaxScaler().fit(X_data.loc[:size])
    X_data=scaler.transform(X_data)

    y_data=labels[i:i+BLOCK_SIZE]
    X=[]
    y=[]
    for z in range(len(X_data)-WIND_SIZ-1):
        X.append(X_data[z:z+WIND_SIZE])
        y.append(y_data[z+WIND_SIZE+1])

    X_train=tf.convert_to_tensor(X[:-2],dtype=tf.float64)
    y_train=tf.convert_to_tensor(y[:-2],dtype=tf.float64)
    X_test=tf.convert_to_tensor([X[-1]],dtype=tf.float64)
    y_test=tf.convert_to_tensor([y[-1]],dtype=tf.float64)

#     print(X_train.shape)
#     print(X_test.shape)    
#     print(y_train.shape)
#     print(y_test.shape)

    model = MODEL

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=tf.keras.metrics.BinaryAccuracy())

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=5, min_lr=0.00001)

    history=model.fit(X_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,
                  validation_data=(X_test,y_test),
                  shuffle=False,verbose=0,
                  use_multiprocessing=True,workers=100,
                  callbacks=[])
    train_accuracy.append(history.history['binary_accuracy'][-1])
    test_accuracy.append(history.history['val_binary_accuracy'][-1])
    print(history.history['binary_accuracy'][-1],history.history['val_binary_accuracy'][-1])

    preds.append(model.predict(X_test)[0])
    y_rights.append(y_test[0])
#     plt.plot(history.history['binary_accuracy'])
#     plt.plot(history.history['val_binary_accuracy'])
#     plt.title('model mean_absolute_error')
#     plt.ylim(-0.1,1.1)
#     plt.xlim(0,EPOCHS+1)
#     plt.ylabel('binary_accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')

#     predictions=model.predict(X_train)
#     plt.figure(figsize=(20, 8))
#     plt.plot(predictions,color='r')
#     plt.plot(y_train)

#print(f'accuracy----------{np.sum([np.array(test_accuracy)==1])/len(test_accuracy)}')

plt.figure(figsize=(6, 5))
plt.plot(train_accuracy)
plt.plot(test_accuracy)
plt.title(f'model binary_accuracy = {np.sum([np.array(test_accuracy)==1])/len(test_accuracy)}')
plt.ylim(-0.1,1.1)
plt.ylabel('binary_accuracy')
plt.xlabel('step')
plt.legend(['train', 'test'], loc='upper left')

plt.figure(figsize=(20, 8))
plt.plot(preds,color='r')
plt.plot(y_rights)
plt.legend(['preds', 'right'], loc='upper left')

############################################################################################################################################################################
BLOCK_SIZE=120
WIND_SIZE=16
MAX_EP=15
MIN_EP=10

BATCH_SIZE=16
FECHES=['body']#'body','del_vol','highlow','del_coin_amount'

print(f'COFIG :: block {BLOCK_SIZE}, wind {WIND_SIZE}, epoches {EPOCHS}, batch {BATCH_SIZE}')

MODEL= tf.keras.models.Sequential([
        tf.keras.layers.LSTM(20,return_sequences=True),
        tf.keras.layers.LSTM(10),
        tf.keras.layers.Dense(10,activation='ReLU'),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])        


train_accuracy=[]
test_accuracy=[]
preds=[]
y_rights=[]

data=full_frame.loc[6500:]
data.index=range(len(data))

labels=[]
for i in data.body:
    if i>0:labels.append(1)
    else:labels.append(0)
        
X_data=data[FECHES].loc[:BLOCK_SIZE-1].to_numpy()
y_data=labels[:BLOCK_SIZE]       
        
for i in range(len(data)-BLOCK_SIZE):
    if(i>9):break
        
    X_data=np.vstack([X_data,data[FECHES].loc[i+BLOCK_SIZE]])
    size=len(X_data)-2
    scaler=MinMaxScaler().fit(X_data[:size])
    X_data=scaler.transform(X_data)

    y_data.append(labels[i+BLOCK_SIZE])
    
    X=[]
    y=[]
    for z in range(len(X_data)-WIND_SIZE):
        X.append(X_data[z:z+WIND_SIZE])
        y.append(y_data[z+WIND_SIZE])

        
#     print(X[-3],y[-3])
#     print(X[-1],y[-1])
    X_train=tf.convert_to_tensor(X[:-2],dtype=tf.float64)
    y_train=tf.convert_to_tensor(y[:-2],dtype=tf.float64)
    X_test=tf.convert_to_tensor([X[-1]],dtype=tf.float64)
    y_test=tf.convert_to_tensor([y[-1]],dtype=tf.float64)

    print(X_train.shape)
    print(X_test.shape)    
    print(y_train.shape)
    print(y_test.shape)
    
    model = MODEL

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=tf.keras.metrics.BinaryAccuracy())

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=19, min_lr=0.00001)
    counter=0
    eps=0
    model.optimizer.learning_rate.assign(0.01)
    
    while(True):  
        model_buf=tf.keras.models.clone_model(model)
        
        history=model.fit(X_train,y_train,batch_size=BATCH_SIZE,epochs=1,
                      validation_data=(X_test,y_test),
                      shuffle=False,verbose=0,
                      use_multiprocessing=True,workers=100,
                      callbacks=[])
        counter+=1
        eps+=1
#         if history.history['binary_accuracy'][-1]>0.9:break
#         elif history.history['binary_accuracy'][-1]>0.7:
#             if counter>MIN_EP:break
#         elif history.history['binary_accuracy'][-1]>0.5:
#             if counter>MAX_EP:break
#         else:
#             if counter>MIN_EP:
#                 model.optimizer.learning_rate.assign(0.001)
#                 counter-=1
        if eps==MAX_EP:break
    print('eps:',eps)
        
    train_accuracy.append(history.history['binary_accuracy'][-1])
    test_accuracy.append(history.history['val_binary_accuracy'][-1])
    print(history.history['binary_accuracy'][-1],history.history['val_binary_accuracy'][-1])

    preds.append(model.predict(X_test)[0])
    y_rights.append(y_test[0])
#     plt.plot(history.history['binary_accuracy'])
#     plt.plot(history.history['val_binary_accuracy'])
#     plt.title('model mean_absolute_error')
#     plt.ylim(-0.1,1.1)
#     plt.xlim(0,EPOCHS+1)
#     plt.ylabel('binary_accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')

#     predictions=model.predict(X_train)
#     plt.figure(figsize=(20, 8))
#     plt.plot(predictions,color='r')
#     plt.plot(y_train)

#print(f'accuracy----------{np.sum([np.array(test_accuracy)==1])/len(test_accuracy)}')

plt.figure(figsize=(6, 5))
plt.plot(train_accuracy)
plt.plot(test_accuracy)
plt.title(f'model binary_accuracy = {np.sum([np.array(test_accuracy)==1])/len(test_accuracy)}')
plt.ylim(-0.1,1.1)
plt.ylabel('binary_accuracy')
plt.xlabel('step')
plt.legend(['train', 'test'], loc='upper left')

plt.figure(figsize=(20, 8))
plt.plot(preds,color='r')
plt.plot(y_rights)
plt.legend(['preds', 'right'], loc='upper left')

#######################################################################################################################################################################################################
print('//////////////////////////////////////////////////////DATA///////////////////////////////////////////////////////')
print('OBSERVATIONS_NUM : ',OBSERVATIONS_NUM)
print('TRAIN_SIZE : ',TRAIN_SIZE)
print('LAST_ACTIVATION : ',LAST_ACTIVATION)
print('LABELS_IS_BYNARY : ',LABELS_IS_BYNARY)
print('SCALER : ',SCALER)
print('WIND_SIZE : ',WIND_SIZE)
print()
print('///////////////////////////////////////////////TRAIN-TEST SHAPES/////////////////////////////////////////////////')
print('X_train:',X_train.shape)
print('y_train:',y_train.shape)
print('X_test:',X_test.shape)    
print('y_test:',y_test.shape)
print()
print('//////////////////////////////////////////////////////MODEL//////////////////////////////////////////////////////')
print('IS_LSTM :',IS_LSTM)
print(MODEL.summary())
print()
print('OPTIMIZER : ',OPTIMIZER,'  LEARNING_RATE : ',LEARNING_RATE)
print('LOSS : ',LOSS.name)
print('METRICS : ',METRICS.name)
print()
print('/////////////////////////////////////////////////////FITTING/////////////////////////////////////////////////////')
print('EPOCHES :',EPOCHES)
print('BATCH_SIZE : ',BATCH_SIZE)
print()
print('//////////////////////////////////////////////////////RESULTS//////////////////////////////////////////////////////')
print('TEST METRIC : ',history.history[f'val_{METRICS.name}'][-1])
print('TRAIN METRIC : ',history.history[METRICS.name][-1])

