You need to create an object of either the TheoreticallyOptimalStrategy and run testPolicy

You need to create an object of either the ManualStrategy and run testPolicy

e.g.
example = ManualStrategy()
example.testPolicy(symbol='JPM', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2010, 6, 1), sv=100000)




To get the charts for the Indicators you can run Indicators from Indicators.py
e.g.
Indicators(symbols=symbols, startDate=start_date, endDate=end_date)

