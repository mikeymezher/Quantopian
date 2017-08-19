
from quantopian.pipeline import Pipeline, CustomFilter
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.factors import Latest
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data.psychsignal import aggregated_twitter_withretweets_stocktwits as st
from sklearn.preprocessing import StandardScaler
#from quantopian.pipeline.data.psychsignal import stocktwits as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor 
#Note that Quantopian uses sklearn v.16.1 - ie MLPRegressor is not available
from quantopian.pipeline.filters import Q500US
from quantopian.pipeline.filters import Q1500US
import numpy as np
from odo import odo 
import pandas as pd
#Basic model that trades based on recent stock twit and twitter sentiment data
def initialize(context): 
   
    context.days_collection=10 #Number of days to collect data before building model (initially used 14)
    context.delLen=0
    context.y_portVal=0
    context.performances=[]
    context.yest_feats=[]
    context.Y_securities=[]
    context.Cs=[]
    context.Os=[]
    context.refit=False
    context.results=[[0],[0]]
    context.yest_res=[]
    context.long_securities=[]
    context.long_matrixResults=[]
    context.long_outcomes=[]
    context.last_predictions=[]
    context.securities_in_results = []
    context.sentimentMultiplier=3.5 #Number of times more bull messages than bear required to make it past scree 
    #context.model = RandomForestRegressor()
    #context.model = RandomForestRegressor(warm_start=True) #******Make sure this model is a regressor*****
    context.model = GradientBoostingRegressor(warm_start=True)
    #context.model = GradientBoostingRegressor()
    context.scaler = StandardScaler()
    #context.model = KNeighborsRegressor()
    #context.model2 = RandomForestRegressor()
    #context.model2 = RandomForestRegressor(warm_start=True)
    #context.model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    #context.model2 = SVR(kernel='linear', C=1e3, gamma=0.1)
    #context.model.n_estimators = 20 #Original val 500
    #context.model.min_samples_leaf = 200 #Original val 100
    context.ran = 0 #Used to build initial model (and subsequent secondary models - every context.days_collection days)
    context.ran2=0
    
    #schedule_function(sellAll, date_rules.every_day(), time_rules.market_close(minutes=5)) #IF we are training our model on a target that has a result (open - close) percent change, we will match this more closely by selling all our securities 5 minutes before market close
    
    schedule_function(build_model,date_rules.every_day(),time_rules.market_open()) 
    schedule_function(trade, date_rules.every_day(), time_rules.market_open())
    schedule_function(calc_performance, date_rules.every_day(), time_rules.market_open())
    set_commission(commission.PerShare(cost=0.0, min_trade_cost=0.0))
    
    attach_pipeline(Custom_pipeline(context), 'sentiment_pipe') 
    
def Custom_pipeline(context):
    set_symbol_lookup_date('2016-1-1')
    #Not a bad list of securities. But when using S&P as baseline it would be unfair to backtest using securities we already know succeed. 
    #my_sid_filter = SidInList(
        #sid_list = (
            #symbol('AMD').sid,symbol('NVDA').sid,symbol('BRK_B').sid,
            #symbol('MU').sid,symbol('IBM').sid,symbol('TRUE').sid,
            #symbol('AMAT').sid,symbol('AMZN').sid,symbol('MSFT').sid,
            #symbol('TSLA').sid,symbol('GLW').sid,symbol('ATVI').sid,
            #symbol('AMAT').sid,symbol('MU').sid,symbol('ETRM').sid,symbol('IDXG'),)) 
           
    
    pipe = Pipeline()
    #TO INCLUDE: Yesterday's data
    
    pipe.add(st.bull_scored_messages.latest, 'bull_scored_messages')
    pipe.add(st.bear_scored_messages.latest, 'bear_scored_messages')
    pipe.add(st.bullish_intensity.latest, 'bullish_intensity')
    pipe.add(st.bearish_intensity.latest, 'bearish_intensity')
    pipe.add(st.total_scanned_messages.latest, 'all_messages')
    
    pipe.set_screen(Q500US() & ( (st.bull_scored_messages.latest) > (context.sentimentMultiplier* st.bear_scored_messages.latest) ) & (st.bear_scored_messages.latest > 10) ) 
    return pipe

#Build a model based on historic sentiment data and that day's security resulting open and close price (This will only be an initial model, another model will be developed in real time and updated daily to contanstly improve its parameters) 

#The question is whether or not to make the security itself (AAPL, TSLA, AMD, etc) a feature of the classifier as well (as in train the classifier to make security specific predictions or train it to make more general predictions). Security specific predictions are good in the sense that the classifier is likely to be able to form a better decision boundary, however - the predictive value may be less since #1 there is less training data, and #2 the outcome of the classifier's decision is likely to be more based on the stock's general trend then the sentiment features.

#Since it isn't possible yet to get historic pipeline data, we'll run the model for 14 days first - in order to compile data to build a sufficient classifier. 
def build_model(context, data):
    if context.ran>context.days_collection:
    	return
    results=pipeline_output('sentiment_pipe')
    for sec in context.Y_securities: #For YESTERDAY'S SECURITIES
        context.long_securities.append(sec)
        yesterO=((data.history(sec,'open',2,'1d')).as_matrix())[0] #Yesterday's open and close
        yesterC=((data.history(sec,'close',2,'1d')).as_matrix())[0]
        todayO=((data.history(sec,'open',2,'1d')).as_matrix())[1]
        
        #yesterC=todayO #Call yesterday's close today's open (instead of switching around all variables)
        #print(yesterO)
        #print(yesterC)
        perDiff=((yesterC-yesterO)/(yesterO))*100#Close price - open price/ open price (percent change)
        #Recall that percent diff is multiplied by 100 here, this will be important if/when updating the classifier****
        #Note also that this is our "Y" (target vector). Since these are YESTERDAY'S values, the X matrix will need to be 0 padded (0 padding rows = number of rows for the FIRST percent difference)
        context.long_outcomes.append(perDiff)
        context.Os.append(yesterO)
        context.Cs.append(yesterC)
    context.Y_securities=results.index #Store yesterday's securities
    
    
    
    matrixResults=results.as_matrix()
    for row in matrixResults:
    	context.long_matrixResults.append(row)   #Long matrix results is essentially our X (feature set)0 
    context.ran+=1
    context.ran2+=1
   
    #print(len(context.long_matrixResults))
    #print(len(context.long_outcomes))
    if context.ran==context.days_collection:
        context.refit=True
    if context.refit: #On the last day of data collection, build the model (and check for refits every day after context.days_collection days
        context.delLen=len(matrixResults) #Number of rows to from START of long_outcomes delete prior to training the classifier (and number of rows to remove from END of X (long_matrixResults))
        #print('Del len: ' + str(context.delLen))
        if len(context.long_outcomes)<len(context.long_matrixResults):
        	X=context.long_matrixResults[:(len(context.long_outcomes)-len(context.long_matrixResults))]
        else:
            X=context.long_matrixResults
        print(len(context.long_outcomes))
        print(len(context.long_matrixResults))
        context.scaler.fit(X)
        X_s = context.scaler.transform(X) #Scaled X values - scaled each column to a mean of .5 
        Y=context.long_outcomes
        Cs=context.Cs
        Os=context.Os
        #print('Verifying lengths' + str(len(context.long_securities)) + ' ' + str(len(X)) + ' ' + str(len(Y)))
        #print('Securities')
        #print(context.long_securities)
        #print('X')
        #print(X)
        #print('Y ')
        #print(Y)
        #print('Os')
        #print(Os)
        #print('Cs')
        #print(Cs)
        print('Mean performance: ' + str(np.mean(context.performances)))
        if np.mean(context.performances)<=0: #If the mean performance over the last context.days_collection days is poor, fit a new model (reset current feature and result sets)
            context.model.fit(X_s,Y) #Fit the model with the data that has been preliminarily collected
            context.long_matrixResults=[]
            context.long_outcomes=[]
            context.refit=False
            print(context.long_securities)
            print(context.Os)
            print(context.Cs)
            print(X)
            context.Y_securities=[]
            context.long_securities=[]
            context.Os=[]
            context.Cs=[]
            print('Fit a new model')
        #print('Model has been fit')
        context.ran=0 #Reset and begining building a new model. 
    	pass
   
    pass

def calc_performance(context, data):
    #Function to calculate current performance of model, and hypothetical performance of potential newly trained model
    portVal=context.portfolio.portfolio_value
    try:
    	todaysPerf=((portVal-context.y_portVal)/(context.y_portVal))*100 #Per change in portfolio value from yesterday till today
        if(context.y_portVal==0):
            todaysPerf=0
        print('Today\'s performance: ' + str(todaysPerf))
    except:
        pass
    context.y_portVal=portVal
    context.performances.append(todaysPerf) #Append today's performance to a list of performances (for performance momentum calculations (store last context.days_collection number of performances)
    if len(context.performances)>context.days_collection:
        context.performances.pop(0) #Note that most recent performance is index [-1]
        
def buildHypeModel(context,data): 
    #this function builds a hypothetical model that will be tested in parallel with the active model. This model will be updated every context.days_collection days. If this model performs better than the present model, we will switch to our present model (then continue to build a new hypothetic model). Note, the temp model is blind (for now) meaning we don't know it's theoretic outcome. Regardless, we will switch to this model if the active model is performing poorly. 
    
    #SCRATCHED FUNCTION. Instead we built this functionality into the original "build_model" function
    
    
    
    
    
    pass
def before_trading_start(context, data):
    record(Leverage = context.account.leverage,pos=len(context.portfolio.positions))
    context.yest_sec=context.securities_in_results #Store yesterday's securities before resetting the matrix
    context.yest_feats=[]
    try:
    	context.yest_feats=context.results.as_matrix() #Store yesterday's features before resetting matrix
    except:
        pass
    context.yest_res=[]
    for sec in context.yest_sec:
        yesterO=((data.history(sec,'open',2,'1d')).as_matrix())[0] #Yesterday's open and close
        yesterC=((data.history(sec,'close',2,'1d')).as_matrix())[0]
        todayO=((data.history(sec,'open',2,'1d')).as_matrix())[1]
        #yesterC=todayO
        perDiff=((yesterC-yesterO)/(yesterO))*100 #Close price - open price/ open price (percent change)
        context.yest_res.append(perDiff)
    
    
    context.results = pipeline_output("sentiment_pipe")
    context.securities_in_results=[]
    
    for s in context.results.index:
        context.securities_in_results.append(s) 
    context.matrixResults=context.results.as_matrix() #Each row corresponds to a single security, each column is a feature
   # print(results.as_matrix())
   # print(context.securities_in_results)
    #if len(context.securities_in_results) > 0.0:                
        #log.info(results)
    
    
def trade (context, data):
    if context.ran2<=context.days_collection:
        return
    #print('Beginning Trade')
    longs= []
    yX,yY=context.yest_feats,context.yest_res #setup for warm-start retraining of model with updated model (inclusive of yesterdays values and results)
    
    ypX=context.last_predictions #These are YESTERDAY'S predicitions (for refeed into a seperate model - this essentially accomplishes having a dynamic decision boundary. (Train a model with yesterday's predictions and the outcome.-Thereby stacking the two models)
    try:
        yX_s = context.scaler.transform(yX) #Scales yesterday's feature set
    	#context.model.fit(yX_s,yY) #Refit the model every day to include yesterday's outcome 
    except:
        print('Model not refit')
        pass
   
    #print('Refit Model') 
    #print('Number of features: ' + str(context.model.n_features_))
    #try:
        #context.model2.fit(ypX,yY) #Refit the second (stacked) model to include yesterday's predictions and outcomes
    #except:
        #print('Model 2 not fit')
        #print(len(ypX))
        #print(len(yY))
        
    context.last_predictions=[]
    if len(context.securities_in_results) > 0:
        for sec in range(len(context.securities_in_results)):
            sec1=context.securities_in_results[sec]
            X=context.matrixResults[sec]
            X_s = context.scaler.transform(X) #Scaler feature set
            prediction=context.model.predict(X_s) #At this point, the model should have been built (on the first day, build the model from historic data, update the model daily based on outcomes)  
            #try:
            	#prediction=context.model2.predict(prediction) #Stacked model prediction (may want to provide seperate variable here* Later, this will be used in conjunction with moving average/ other technical and fundamental based predictors to stack together multiple seperate classifiers making predictions on seperate data.
                #print('Model 2 fit')
            #except:
                #pass
            #print('pred: ' + str(prediction))
            context.last_predictions.append(prediction)
            if(prediction>1): #If the stock is predicted to go up more than 1 percent that day
            	print(str(sec1.symbol) +  " | " + str(prediction))
            	if sec not in context.portfolio.positions:
                	longs.append(sec1)
                
    open_orders=get_open_orders()
    for sec1 in context.portfolio.positions:
        if sec1 not in longs:
            order_target_percent(sec1, 0.0)
    for sec1 in longs:
        if sec1 not in open_orders:
            order_target_percent(sec1,1.0/len(longs))
            #Immediately put a stop market order (if the security falls more than one percent, sell the security (this is meant to prevent hold a stock through an entire missed earnings day or other similar situation))
            stopPrice=(data.current(sec1,'price'))-(.02*data.current(sec1,'price')) #2 percent less than the current price
            #order_target_percent(sec1,0.0,style=StopOrder(stopPrice))
                    
            
#TODO: 
#"Warmstart" training of classifier (to ensure that initial lookback time doesn't completely dictate the classifier's behavior, since the dataset is relatively small)  
    
            
def sellAll(context,data):
    for pos in context.portfolio.positions:
        order_target_percent(pos,0)
