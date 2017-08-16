from quantopian.pipeline import Pipeline, CustomFilter
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.factors import Latest
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data.psychsignal import aggregated_twitter_withretweets_stocktwits as st
#from quantopian.pipeline.data.psychsignal import stocktwits as st
from sklearn.ensemble import RandomForestRegressor
from quantopian.pipeline.filters import Q500US
from quantopian.pipeline.filters import Q1500US
import numpy as np
from odo import odo 
import pandas as pd
#Basic model that trades based on recent stock twit and twitter sentiment data
def initialize(context): 
    context.days_collection=14 #Number of days to collect data before building model
    context.delLen=0
    context.yest_feats=[]
    context.results=[[0],[0]]
    context.yest_res=[]
    context.long_securities=[]
    context.long_matrixResults=[]
    context.long_outcomes=[]
    context.securities_in_results = []
    context.sentimentMultiplier=2.5 #Number of times more bull messages than bear required to make it past scree 
    context.model = RandomForestRegressor(warm_start=True) #******Make sure this model is a regressor*****
    #context.model.n_estimators = 20 #Original val 500
    #context.model.min_samples_leaf = 200 #Original val 100
    context.ran = 0 #Used to build initial model
    schedule_function(build_model,date_rules.every_day(),time_rules.market_open()) #This will only run once
    schedule_function(trade, date_rules.every_day(), time_rules.market_open())
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
    
    pipe.set_screen(Q1500US() & ( (st.bull_scored_messages.latest) > (context.sentimentMultiplier* st.bear_scored_messages.latest) ) & (st.bear_scored_messages.latest > 3) ) 
    return pipe

#Build a model based on historic sentiment data and that day's security resulting open and close price (This will only be an initial model, another model will be developed in real time and updated daily to contanstly improve its parameters) 

#The question is whether or not to make the security itself (AAPL, TSLA, AMD, etc) a feature of the classifier as well (as in train the classifier to make security specific predictions or train it to make more general predictions). Security specific predictions are good in the sense that the classifier is likely to be able to form a better decision boundary, however - the predictive value may be less since #1 there is less training data, and #2 the outcome of the classifier's decision is likely to be more based on the stock's general trend then the sentiment features.

#Since it isn't possible yet to get historic pipeline data, we'll run the model for 14 days first - in order to compile data to build a sufficient classifier. 
def build_model(context, data):
    if context.ran>context.days_collection:
    	return
    results=pipeline_output('sentiment_pipe')
    for sec in results.index:
        context.long_securities.append(sec)
        yesterO=((data.history(sec,'open',2,'1d')).as_matrix())[0] #Yesterday's open and close
        yesterC=((data.history(sec,'close',2,'1d')).as_matrix())[0]
        #print(yesterO)
        #print(yesterC)
        perDiff=((yesterO-yesterC)/(yesterO))*100#Close price - open price/ open price (percent change)
        #Recall that percent diff is multiplied by 100 here, this will be important if/when updating the classifier****
        #Note also that this is our "Y" (target vector). Since these are YESTERDAY'S values, the X matrix will need to be 0 padded (0 padding rows = number of rows for the FIRST percent difference)
        context.long_outcomes.append(perDiff)
    if context.ran==0: 
        context.delLen=len(context.long_outcomes) #Number of rows to from START of long_outcomes delete prior to training the classifier (and number of rows to remove from END of X (long_matrixResults))
    
    matrixResults=results.as_matrix()
    for row in matrixResults:
    	context.long_matrixResults.append(row)   #Long matrix results is essentially our X (feature set)0 
    context.ran+=1
    
    #print(len(context.long_matrixResults))
    #print(len(context.long_outcomes))
    if context.ran==context.days_collection: #On the last day of data collection, build the model
        X=context.long_matrixResults[:-context.delLen]
        Y=context.long_outcomes[context.delLen:]
        print('X ')
        print(X)
        print('Y ')
        print(Y)
        context.model.fit(X,Y) #Fit the model with the data that has been preliminarily collected
        print('Model has been fit')
    	pass
   
    pass

def before_trading_start(context, data):
    record(Leverage = context.account.leverage,pos=len(context.portfolio.positions))
    context.yest_sec=context.securities_in_results #Store yesterday's securities before resetting the matrix
    try:
    	context.yest_feats=context.results.as_matrix() #Store yesterday's features before resetting matrix
    except:
        pass
    for sec in context.yest_sec:
        yesterO=((data.history(sec,'open',2,'1d')).as_matrix())[0] #Yesterday's open and close
        yesterC=((data.history(sec,'close',2,'1d')).as_matrix())[0]
        perDiff=((yesterO-yesterC)/(yesterO))*100 #Close price - open price/ open price (percent change)
        context.yest_res.append(perDiff)
    
    context.results = pipeline_output('sentiment_pipe')
    context.securities_in_results=[]
    
    for s in context.results.index:
        context.securities_in_results.append(s) 
    context.matrixResults=context.results.as_matrix() #Each row corresponds to a single security, each column is a feature
   # print(results.as_matrix())
   # print(context.securities_in_results)
    #if len(context.securities_in_results) > 0.0:                
        #log.info(results)
    
    
def trade (context, data):
    if context.ran<=context.days_collection:
        return
    #print('Beginning Trade')
    longs= []
    yX,yY=context.yest_feats,context.yest_res #setup for warm-start retraining of model with updated model (inclusive of yesterdays values and results)
    context.model.fit(yX,yY)
    #print('Refit Model') 
    #print('Number of features: ' + str(context.model.n_features_))
    if len(context.securities_in_results) > 0:
        for sec in range(len(context.securities_in_results)):
            sec1=context.securities_in_results[sec]
            X=context.matrixResults[sec]
            prediction=context.model.predict(X) #At this point, the model should have been built (on the first day, build the model from historic data, update the model daily based on outcomes)  
            print('pred: ' + str(prediction))
            if(prediction>.35): #If the stock is predicted to go up more than .35 percent that day
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
            
#TODO: 
#"Warmstart" training of classifier (to ensure that initial lookback time doesn't completely dictate the classifier's behavior, since the dataset is relatively small)  
    
            

