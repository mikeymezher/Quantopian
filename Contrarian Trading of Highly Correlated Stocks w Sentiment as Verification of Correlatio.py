from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import CustomFactor, Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline.filters.morningstar import Q1500US
from quantopian.pipeline.filters.morningstar import Q500US
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.classifiers.morningstar import Sector
import numpy as np
from scipy import signal
import itertools as itools
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from quantopian.pipeline.data.psychsignal import aggregated_twitter_withretweets_stocktwits as st


def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    
    
    
    context.Sentiment_multiplier = 1.7 #Number of times more bull messages than bear messages a stock must have had to be acceptable for trading
    # Create our dynamic stock selector.
    attach_pipeline(make_pipeline(context), 'my_pipeline')
    context.minCorr=0.2 #Minimum acceptable cross correlation value to make trades based on
    context.minCorr_short=.5 #Min acceptable cross correlation value (for short term tau) 
    
    context.lookback=1800 #This is the number of minutes used during cross correlation of data. Note, that this should be less than 370 (390 trading day minutes - 20 min end of day stop) since we correlate same day data then trade based on those correlation. (Correlate first half the day, trade second half) 
    # *****^ Same day correlation may be the best way to figure out tau, but it may be suboptimal for selecting commonly correlated stocks. This selection is more likely to make errors in picking stocks which have a coincidentally high correlation (just for one particular day). 
    context.shortTau=180
 
#Note, it may be ideal to select a tau based on the correlation method described below (just using the early half of the day) and to select likely correlated stocks seperately using a longer lookback time. ***
#**********TODO*********************************************************
    #Methods of adding value to correlation:
    #Since we are trading on spike patterns only, it could be in our best interest to square (or even cube) the entire time series of a stock before normalizing it for (normalized) cross correlation. This would serve to exaggerate deviant values (high peaks). However, this would also underexaggerate drops, which we are interested in correlating as well. THEREFORE:
    #Correlate stocks by:
    #1.) 0 center the mean of the stock's time series. 
    #2.) take the absolute value of the time series. - WHILE PRESERVING THE LOCATION (IDXS) OF NEGATIVE VALUES 
    #3.) Square (or cubed the entire time series (thereby exaggerating both the peaks and drops in the time series.
    #4.) Reverse the absolute value transformation with the exagerated peaks and drops. 
    #5.) Normalize and correlate. 
#**********************************************************************
    
    context.slope_lookback=20#Number of minutes used when determining sudden price change. 
    context.perChange = .07 #Percent change over length of 1/3(context.slope_lookback) that would indicate a sudden price change (This is the percent change that trades will be made on) **May be best adjusted based on volatility***
    context.numTradeCombo=5 #Number of most correlated combinations we wish to keep.
    #TODO: lookback may be best adjusted daily, to account for long term change in ideal lookback time.
    context.timerList=[] #List of times of when to make a trade (unit in minutes from present)
    context.tradeList=[] #List of securities to trade 
    context.actionList=[] #List of actions (buy = 1  default (no action) = 0 sell = -1)
    context.tradingNow=[] #List of whether or not a security pair is currently trading (abrupt change already detected (yes = 1 or no = 0)
    context.tempCash=0
    
    context.baseThresh=0 #Percent change used as a threshold for determining the time of the base of a spike. Also used to verify the presence of an occuring spike on the lag stock prior to purchase  
    context.fallThresh=-.0001 #Rate at which fall has to occur before an end of spike has been indicated (Usually 0)
    
    #Sell all securities at the end of each trading day (5 minutes before close of market)
    schedule_function(sellAll, date_rules.every_day(), time_rules.market_close(minutes=15))
    #Coordinate trades every minute. in the given range (minTime). Note that half days will not be traded and that the end of the range is 370 minutes after market open (369 - ie 21 minutes prior to market close)
    
    endDay=185 #Time to stop correlating and start trading (lets half a trading days worth of correlation be included)
    startTrade=endDay+2
    endTrade=370
    
    for minTime in range(startTrade,endTrade,1):
    	schedule_function(coordTrade, date_rules.every_day(), time_rules.market_open(minutes = minTime), half_days=False)
    schedule_function(getIdealSec, date_rules.every_day(), time_rules.market_open(minutes = endDay))
    set_commission(commission.PerShare(cost=0.0, min_trade_cost=0.0))   

           
def make_pipeline(context):
    """
    A function to create our dynamic stock selector (pipeline). Documentation on
    pipeline can be found here: https://www.quantopian.com/help#pipeline-title
    """
    
    # Base universe set to the Q1500US
    base_universe = Q500US()
    
    #Get all industry codes
    industry=morningstar.asset_classification.morningstar_industry_code.latest
    #Get all sector codes
    sector = Sector()
    
    # Create filters (to be used as masks) of different industries/sectors 
    # This is the mask that should exclude the most stocks. 
    # Note that these may need to be even further filtered to exclude securities outside of a 
    # similar range of volumes/size. For instance, the defense sector stock provides stocks as large as     # LMT but also small defense companies. Although this shouldn't matter due to the second filter of 
    # crosscorrelation, this may be unnecassary computational expense. 
    pipe=Pipeline()
    #Below forms a "sentiment screen" that takes only stocks that have been rated a certain number of times and of those ratings there are at least 2.85 times as many bull scored messages as there are bear scored messages. 
    pipe.add(st.bull_scored_messages .latest, 'bull_scored_messages')
    pipe.add(st.bear_scored_messages .latest, 'bear_scored_messages')
    sentimentScreen=(((st.bull_scored_messages.latest) > (context.Sentiment_multiplier*st.bear_scored_messages.latest)) & (st.bear_scored_messages.latest > 5))
    
    dFilt=sector.eq(310) #Indicates aerospace/defense sector
    dFilt2=industry.eq(31052107) #Indicates aerospace/defense industry
    tFilt=sector.eq(311) #Indicates consumer electronics sector
    tFilt2=industry.eq(31167138) #Indicates consumer electronics industry 
    cFilt=sector.eq(101) #Chemical sector
    cFilt2=industry.eq(10103003)
    aFilt=sector.eq(102)
    aFilt2=industry.eq(10209017) #Auto manufacturing industry
    depFilt2=industry.eq(10217034) #Department store industry
    #dFilt2,tFilt2,cFilt2,aFilt2=True,True,True,True #Remove industry requirement
    defenseFilt= dFilt & dFilt2  #Combination of filters
    techFilt= tFilt & tFilt2
    chemFilt = cFilt & cFilt2 
    autoFilt = aFilt & aFilt2  
    tradable=base_universe & (defenseFilt | techFilt | chemFilt | autoFilt | depFilt2) & sentimentScreen
    
    
    pipe.set_screen(tradable)
    pipe.add(defenseFilt,'defenseFilt')
    pipe.add(techFilt,'techFilt')
    pipe.add(chemFilt,'chemFilt')
    pipe.add(autoFilt,'autoFilt')
    pipe.add(depFilt2,'depFilt')
        
        
    
    #TODO: May also want to return stock sentiment data and further filter tuple couples by only accepting couples with sentiment data in a similar range (further attributing to the validity of the calculated cross-correlation)
  
    return pipe

 
def getIdealSec(context, data): #This replaced before_trading_start(context, data)
    """
    Called every day before market open.
    """
    record(Leverage = 																					           context.account.leverage,pos=len(context.portfolio.positions))
    context.output = pipeline_output('my_pipeline')
    #print('Pipeout: ')
    #print(context.output)
  
    # These are the securities that we are interested in trading each day.
    # Note: As it stands, the securities in this list are from two different industries (defense and
    # consumer electronics). Although more computationally expensive then dividing them out into their 
    # two respective industries prior to cross correlating, leaving them in the same matrix/data set and 
    # cross correlating them gives us a way to 'check' that the crosscorrelation is valid, since               securities within the same industry should typically cross correlate to a higher degree than across industries. ***
    context.security_list = context.output.index 
    context.defenseList = context.output[context.output['defenseFilt']].index.tolist()
    #print(context.defenseList)
    context.autoList = context.output[context.output['autoFilt']].index.tolist()
    #print(context.autoList)
    context.chemList = context.output[context.output['chemFilt']].index.tolist()
    #print(context.chemList)
    context.techList = context.output[context.output['techFilt']].index.tolist()
    #print(context.techList)
    context.depList = context.output[context.output['depFilt']].index.tolist()
     # Within each sector, calculate the mean (and max, since we may choose only to trade the maximally        correlated securities regardless of industry) crosscorrelation between all combinations of stocks. 
    #This will only run every trading day to prevent computational expense. In that 
    #respect, performs identically to a pipeline add-on (but allows the use of "history") 
    #Try block here incase pipe returns no valid securities. 
    try:
    	price_history = np.transpose(data.history(context.security_list, fields="price",                                         bar_count=context.lookback,frequency="1m"))
    	price_history=price_history.as_matrix()
    except:
        price_history=[[0],[0],[0]]
    #This returns three arrays, containing a filtered set of maximally cross correlated securities            within the last time range (given by context.lookback), their associated (and filtered) time delays      corresponding to their maximum correlation, and the degree of their correlation in the given time        frame. Essentially, since tau has already been filtered for, the degree of their correlation should      be used as a confidence feature to make predictions off of, and tau should be used to determine when to make purchases/sales. 
    #hCorrVals,maxSecs,timeDelays,short_timeDelays=crossCorr(context.security_list,price_history,context)
    #The best securities to trade using this algorithm (each day) are listed in the below lists ***
    try:
    	hCorrVals,maxSecs,timeDelays,short_timeDelays=crossCorr(context.security_list,price_history,context)   
    except: 
        print('Crosscorr Failed')
        maxSecs,hCorrVals,timeDelays,short_timeDelays=[],[],[],[]
    #"Globalize" the returned information so that we can handle these commodities every minute. 
    context.Securities=maxSecs
    context.CorrVals=hCorrVals
    context.timeDelays=short_timeDelays #************Used to be timeDelays, now however, we calculate a more recent tau
    context.actionList,context.timerList,context.tradeList,context.tradingNow=[0]*len(context.Securities),[0]*len(context.Securities),[0]*len(context.Securities),[0]*len(context.Securities) #list of zeros indicating that no stocks should currently be trading
    #(Note that all stocks should be sold at end of every tradinng day.) 
  
   
#TODO: Compute cross correlation for 1/3(context.lookback) and 2/3(context.lookback) as additional confidence features. (Ie if the correlation was high in all 3 time segments (given that there is overlap), the correlation will continue to be high the next day. 
#In the same respect, it may also be beneficial to store the last logged time frame correlation results, and use those as additional features 
#The premise of both of these additional potential features is that extended time will likely end up being weighted in a classifier or regressor to a lesser degree than more recent time correlation values.

        

def crossCorr(securities,history,context):
     highCorrVal=0 #Initialize some comparator values
     #Initialize empty lists to return.
     highCorrVals=[]
     highCorrArray=[]
     maxSecs=[]
     timeDelays=[] 
     short_timeDelays=[]
     numSec=len(history) #Number of securities being evaluated
     combinations=list(itools.combinations(range(numSec),2)) #All possible combinations of securities

     for c in range(len(combinations)):
        negValIdxs=[]
         #Here we get the history of one stock (to cross correlated with the next)
        uN_secHist1=history[combinations[c][0]]
        #ADDED FUNCTIONALITY: Cubing of rectified time series to exagerate deviations prior to normalized correlation.
       
        #1.) 0 the mean:
        histMean=np.mean(uN_secHist1)
        uN_secHist1=np.subtract(uN_secHist1,histMean) #Subtract mean from all vals in array
        #print('Zero centered')
        #2.) Note the location of all negative values
        for ii in range(len(uN_secHist1)):
            if uN_secHist1[ii] < 0:
                negValIdxs.append(ii)
        #print('Got neg idxs')
        #3.) Rectify the signal (take the absolute value)
        uN_secHist1=abs(uN_secHist1)
        #print('Rectified')
        #4.) Exponate the signal (to exaggerate peaks and falls)
        uN_secHist1=np.power(uN_secHist1,2) #Square each value 
        #print('Powered')
        #5.) Return values to being negative
        for negIdx in negValIdxs:
            uN_secHist1[negIdx]=uN_secHist1[negIdx]*-1
        #print('Unrectified')
        #Finally, proceed with the original correlation method. Repeat for second security as well. 
           
        #Normalize securities being corrrelated so that their correlation is scale independent
        #Scale will be between -1 and 1 normalized ** fix this ** (scaled) over the last X minutes of          trading
        
        negValIdxs=[]
        #We scale the price of the stock such that its mean price is 0.
        scaler = StandardScaler()
        scaler.fit(uN_secHist1)
        secHist1=scaler.transform(uN_secHist1)
        #Then we scale its price further such that it's maximum value is 1 and its minimum is -1
        #Note that at this point, its mean is no longer necessarily 0.
        normScaler=MinMaxScaler(feature_range=(-1, 1))
        normScaler.fit(secHist1)
        secHist1=normScaler.transform(secHist1)
        
        
        #Repeat this process for the security which we are correlating against
        uN_secHist2=history[combinations[c][1]]
        #1.) 0 the mean:
        histMean=np.mean(uN_secHist2)
        uN_secHist2=np.subtract(uN_secHist2,histMean)
        #2.) Note the location of all negative values
        for ii in range(len(uN_secHist2)):
            if uN_secHist2[ii] < 0:
                negValIdxs.append(ii)
        #3.) Rectify the signal (take the absolute value)
        uN_secHist2=abs(uN_secHist2)
        #4.) Exponate the signal (to exaggerate peaks and falls)
        uN_secHist2=np.power(uN_secHist2,2) #Square each value
        #5.) Return values to being negative
        for negIdx in negValIdxs:
            uN_secHist2[negIdx]=uN_secHist2[negIdx]*-1
            
        scaler.fit(uN_secHist2)
        secHist2=scaler.transform(uN_secHist2)
        normScaler.fit(secHist2)
        secHist2=normScaler.transform(secHist2)
        
        shortHist1=secHist1[-1:len(secHist1)-context.shortTau:-1] #Get more recent tau to make predictions based off of. 
        shortHist2=secHist2[-1:len(secHist2)-context.shortTau:-1]
        corrVals=np.correlate(secHist1,secHist2,'same')
        short_corrVals=np.correlate(shortHist1,shortHist2,'same')
        #Finally, we normalize the cross correlation such that it has a min value of -1 and a max of 1
        corrVals=corrVals/len(corrVals) 
        maxCorrVal=max(corrVals)
        short_maxCorrVal=max(short_corrVals)
        #If the index max is small, the first security is leading (the smaller, the more lead)
        idxMax=np.argmax(corrVals) #Maximum correlation value index, will be used to calculate tau.
        short_idxMax=np.argmax(short_corrVals)
        midPoint=context.lookback/2
        short_midPoint=context.shortTau/2
        #Tau (the lag) is calculated below. If tau is negative, the second security leads
        #(The greater the magnitude, the more lag)
        tau=midPoint-idxMax #Tau is given in units of minutes of lag
        short_tau=short_midPoint-short_idxMax
        #Filter also by tau range, can't trade stocks based on no delay correlation, and if delay is too          high, correlation is likely coincidental. Also, correlation must be above minimum acceptable correlation
        #Also verify that all returned security pairs are within the same industry (discard pair that are in different industries. (Call this conditional same_Pipe)
        s1 = securities[combinations[c][0]]
        s2 = securities[combinations[c][1]]
        #Note, this is meant to check if pairs are within the same sector and industry. 
        same_Pipe = ((s1 in context.defenseList and s2 in context.defenseList) or (s1 in context.autoList and s2 in context.autoList) or (s1 in context.chemList and s2 in context.chemList) or (s1 in context.techList and s2 in context.techList) or (s1 in context.depList and s2 in context.depList))
        
        #If we wish to skip over the same_Pipe (same industry filter), simply mark it as true (uncomment below.)
        #same_Pipe=True
        
        if maxCorrVal>=highCorrVal and abs(short_tau)>0 and abs(short_tau)<15 and maxCorrVal >= context.minCorr and short_maxCorrVal >= context.minCorr_short and same_Pipe:
            #Note that these tuple pairs should be checked to ensure the stocks are from the same                    industry (if they aren't there's likely a higher chance the correlation is coincidental
            maxSecs.append((securities[combinations[c][0]],securities[combinations[c][1]]))
            highCorrVals.append(maxCorrVal) #Will need to get index of maximum correlation values (tau)
            highCorrArray.append(corrVals) #New vals added to end of list
            timeDelays.append(tau)
            short_timeDelays.append(short_tau)
            #Values with a loweest degree of correlation are popped from the front of the list
            if len(highCorrVals)>context.numTradeCombo:
                highCorrVal=min(highCorrVals)
                idxMin=highCorrVals.index(min(highCorrVals)) 
                maxSecs.pop(idxMin)
                highCorrVals.pop(idxMin) 
                highCorrArray.pop(idxMin)
                timeDelays.pop(idxMin)
                short_timeDelays.pop(idxMin)
     
     
     #Return a list of highly correlated securities (order matters here), a list of corresponding cross        correlation metrics and a list of corresponding time delays. 
    
     return highCorrVals,maxSecs,timeDelays,short_timeDelays
    
    #TODO: Calculate tau for each tuple pair
    #      Identify securities with a high correlation in multiple instances, especially if they have 
    #      a leading "tau" (that is the maximum correlation time shift is negative with respect to their            correlary (their price change leads)
    #      "Tau" should also be used as a check to ensure correlation will likely hold. Tau should be in            a given range (likely between 2-10 minutes... ***VALIDATE THIS ASSUMPTION***)
    
    #More cross correlation functions, correlating only significant spikes or drops in price, and            sentiment data
                
            
     #Will use numpy.correlate function. Look at conv function documentation. 'same' setting will likely      #work optimally, tau can be defined in that case as the index of the maximum value in the returned      #correlation array. *****    
    
    #NOTE: We could also return securities that are highly INVERSELY correlated, and make predictions       # off the inverse cross correlation 

    
def sellAll(context,data):
    for pos in context.portfolio.positions:
        order_target_percent(pos,0)
        
            
    
#This function is meant to detect abrupt changes in a securities spike (specifically the securities that have been selected to monitor based on the cross correlation values). 
def spikeDetect(recentHist):
    #First we need to define what a sudden change is. One such definition could be any period where the magnitude of the percent change of the last X (X here defined as context.slope_lookback) values is greater than a slope of Y.
    #To determine this, this function simply calculates the percent change of the last X prices
    #Calculating the mean percent change below: (given as normalized 0-1 value ie .1=10% change)
    erRange=len(recentHist)//3 
    erRange=3#Allows for find base to search more extensive history range while still looking for very abrupt price change as a trigger
    midRange=erRange*2
    endRange=erRange*3
    #Er_perChange ("early percent change") is the percent change of the most recent third of the stock price. Mid_perChange ("Middle percent change") is the percent change of the middle third (second of three time "windows") of the stock price. And Base_perChange is the percent change of the last third. 
    #Obviously, the tot_perChange is representative of the percent change over the entire given time window
    
    #Ie: The sum of Er Mid and Base perChange should be equal to tot_perChange**********
    Er_perChange=(recentHist[-1]-recentHist[-(erRange)])/(recentHist[-(erRange)])
    Mid_perChange=(recentHist[-(erRange)]-recentHist[-(midRange)])/(recentHist[-(midRange)])
    Base_perChange=(recentHist[-(midRange)]-recentHist[-(endRange)])/(recentHist[-(endRange)])
    tot_perChange=(recentHist[-1]-recentHist[0])/(recentHist[0]) 
    return Er_perChange,Mid_perChange,Base_perChange,tot_perChange
    pass

def findBase(recentHist,changeThreshold):
    #After the initial detection of a spike, this function goes back and retroactively looks for the time at which the base of the spike occured. 
    winLen=1 #Number of minutes in which between slope calculation will be made.
    #Iterate through (starting with most recent history going backwards) until a slope that doesn't meet the lower threshold is found.  
    baseTime=len(recentHist) #If the base time isn't found, assume the base started prior to the current window
    for wIdx in range(1,len(recentHist),1):
        change=(recentHist[-wIdx]-recentHist[-1*(wIdx+winLen)])/(recentHist[-1*(wIdx+winLen)])
        if change<(changeThreshold/100):
            #Mark this as the time the spike started.
            baseTime=wIdx
            break
    return baseTime #The value returned represents the amount of time before the spike was detected that the base of the spike occured (in minutes)
    
    pass
#Once the timer has come to 0 (assumed lag time has been met) make a trade - ie buy or sell the security
#Note, that trades are always made on the LAGGING securities. 
def makeTrade(data,context,idx,verified):
    openOrders=get_open_orders()
    if data.can_trade(context.tradeList[idx]):
        print('Trade based on: ' + str(context.Securities[idx]) + ' ' + str(context.CorrVals[idx]) + ' ' + str(context.timeDelays[idx]))
        print('Action list: ' + str(context.actionList) + ' Active list: ' + str(context.tradingNow) + ' Index: ' + str(idx))
        
        #Sell all of the indicated security
        if context.actionList[idx]==-1: #and context.tradeList[idx] in context.portfolio.positions:
            order_target_percent(context.tradeList[idx],0)
            context.tradingNow[idx]=0 
            context.actionList[idx]=0
            context.timerList[idx]='Hold' #Indicate sold
            print('Sold ' + str(context.tradeList[idx]))
            
        #Buy a portion of the security proportional to the security's presence in the security list (the percent of portfolio in this increase every time this security is purchased (in contrast to order_target_percent)
        
        if (context.actionList[idx]==1 and (context.tempCash >(context.portfolio.portfolio_value*float(1/(2*float(context.numTradeCombo))))) and context.tradeList[idx] not in openOrders and (context.portfolio.portfolio_value>0) and verified): #and (len(openOrders)==0)):         			
            order_value(context.tradeList[idx],context.portfolio.portfolio_value*float(1/(2*float(context.numTradeCombo))))
            #If purchase is not made for one of the above reasons (if conditionals), then indicate that this correlated couple is not trading. 
            context.timerList[idx]=9999999 #Reset to high value to prevent repeat buying
            print('Purchased ' + str(context.tradeList[idx]))
            context.tempCash-=context.portfolio.portfolio_value*float(1/(2*float(context.numTradeCombo)))
            print('value of purchase: ' + str(context.portfolio.portfolio_value*float(1/(2*float(context.numTradeCombo)))))
        else:
            #context.tradingNow[idx]=0
            print('Num open orders: ' + str(len(openOrders)))
        
    pass
     
#This function is meant to verify the predicted spike IS occuring on the lagging stock prior to purchasing at the perceived base.
def verifySpike(recentHist,context): 
    lookback_range=1 #Minutes to calculate change of slope. (1 default)
    RecentSlope=(recentHist[-1]-recentHist[-(1+lookback_range)])/(recentHist[-(1+lookback_range)])
    if RecentSlope>context.baseThresh:
        return True
    else:
        return False
    pass

def coordTrade(context,data):
    """
    Called every minute.
    """
    context.tempCash=context.portfolio.cash
    for idx in range(len(context.Securities)):
        erSpike,midSpike,baseSpike,totSpike=0,0,0,0
        #For each pair of highly correlated values, get the last X minutes of price data (X defined as context.slope_lookback)
        pH1=np.transpose(data.history(context.Securities[idx][0], fields="price",                                         bar_count=context.slope_lookback,frequency="1m"))
        pH1=pH1.as_matrix() 
        pH2=np.transpose(data.history(context.Securities[idx][1], fields="price",                                         bar_count=context.slope_lookback,frequency="1m"))
        pH2=pH2.as_matrix()
        #Get the tau (time shift) value for this pair of securities
        #RECALL: Positive tau means first security leads, negative means second leads
        tau=context.timeDelays[idx]
        #We COULD detect spikes in BOTH the first and second securities, regardless of tau and then use them as an additional confidence feature (ie if the LAGGING security has an abrupt change in price detected prior to the LEADING security, then further diminish confidence that this pair is a good pair to make acute trades on) ******* 
        if tau>0: 
            lead=pH1 
            lag=1
            lag_hist=pH2
        if tau<0:
            lead=pH2 
            lag=0
            lag_hist=pH1
            
        erSpike,midSpike,baseSpike,totSpike=spikeDetect(lead)
        #NEW SPIKE DETECTION AND TRADE COORDINATION METHOD:
#1.) Use erSpike to detect the actual occurance of a spike (this will be the high threshold)
#2.) Iterate through the last X minutes of historic data of the lead stock, and determine when the BASE of the spike occured. Note, that the slope of the base of the spike will be a lower threshold than originally met to detect the occurance of a spike. 
#3.) Call the time the base is detected Tb. The time we will purchase the lag stock will be Tb+tau. Note that Tb+tau may have already passed, if it has, purchase the stock immediately. 

#This lets us retain the ability to acutely detect only fairly dramatic spikes, while also providing a way to detect where the spike initially occured (rather than find it only after its slope has significantly increased, where it may be ending shortly).


    
#TODO: New methods of initial spike detection (detect more gradual rises as well (use totSpike?)
        if erSpike>(context.perChange/100) and context.tradingNow[idx] == 0:
                #Trade correlated stock in tau minutes. Mark that this security pair has been identified as tradable    
            
            base_Shift=findBase(lead,context.baseThresh) #Find the start of the spike and adjust timer appropriately
            print('Base shift: ' + str(base_Shift))
            
            if tau>0:
                adjustedTau=tau-base_Shift
                if adjustedTau<0:
                    adjustedTau=0 #If the base was detected at some point past tau lag, buy the stock immediately
            if tau<0: 
                adjustedTau=tau+base_Shift
                if adjustedTau>0:
                    adjustedTau=0
                    
            context.timerList[idx]=abs(adjustedTau)
            context.tradeList[idx]=context.Securities[idx][lag]
            context.actionList[idx]=1 
            context.tradingNow[idx]=1
                    
        if erSpike<context.fallThresh and context.tradingNow[idx]==1 and tau!='Hold' and context.actionList[idx] == 0: #If the spike ends, we need to sell the security 
            print('Neg spike conditions met')
            context.timerList[idx]=abs(tau)
            context.tradeList[idx]=context.Securities[idx][lag]
            context.actionList[idx]=-1
            
            
        if context.tradingNow[idx]==1:
                context.timerList[idx]=context.timerList[idx]-1 #Subtract 1 from the trade countdown timer
                #If the timer is -1, it is time to make the trade. 
                if context.timerList[idx]<=-1 and context.actionList[idx]!=0:
                    verified=False
                    if context.actionList[idx]==1:
                        verified=verifySpike(lag_hist,context)
                    makeTrade(data,context,idx,verified) #Calling this function makes the a trade (buy or sell) Note that this function also resets the actionList of any given index (sets it to default value 0 after making a buy (1) or sell (-1)
                    context.actionList[idx]=0 #Note, that since we prevent the algorithm from buying additional stocks while there are open orders, having the action list reset to 0 here will prevent a backlogged spike train from ever trading.
                    	
#Reset the stock's trading status only once the stock has been sold
					
                    #Solution: Reset the tradingNow[idx] in cases where stocks were not purchased due to a backlog. 
                     
            
        #Note that these spikes could be used as features into a learning algo to determine whether or not to make a trade. 
        
        #For now, to prevent the need to train an algo, we'll trade on the premise that if the erspike exceeds some constant threshold then trade the related security in tau minutes. Furthermore, if the er spike is less than the mid spike/base spike, then we know the slope of the spike we are trading is starting to deminish so we should sell in tau minutes.        
      
    # Within each top crosscorrelated security, compute tau (lag factor). (Note that tau should be used       as an additional feature, if the lag is unreasonably long, dont trade, if it's less than 2 min or       so, that combination of stocks might also be difficult to trade on.)
    
    # Optional: Compare sentiment data between top cross correlated stocks, based on the hypothesis that     # highly correlated stocks will continue to be highly correlated if their sentiment is similar.
    #1170 minutes is selected (3 full trading days) 
   
    pass


    

