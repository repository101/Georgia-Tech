import datetime

import numpy as np
import pandas as pd

from optimization import get_portfolio_value, optimize_portfolio
from util import get_data


def str2dt(strng):
    # From grade_optimization.py
    year, month, day = map(int, strng.split("-"))
    return datetime.datetime(year, month, day)


def tests_1():
    # Assess the portfolio
    from collections import namedtuple, OrderedDict

    PortfolioTestCase = namedtuple('PortfolioTestCase', ['inputs', 'outputs', 'description'])
    portfolio_test_cases = [
        PortfolioTestCase(
            inputs=dict(
                start_date='2010-01-01',
                end_date='2010-12-31',
                symbol_allocs=OrderedDict([('GOOG', 0.2), ('AAPL', 0.3), ('GLD', 0.4), ('XOM', 0.1)]),
                start_val=1000000),
            outputs=dict(
                cum_ret=0.255646784534,
                avg_daily_ret=0.000957366234238,
                sharpe_ratio=1.51819243641),
            description="Wiki example 1"
        ),
        PortfolioTestCase(
            inputs=dict(
                start_date='2010-01-01',
                end_date='2010-12-31',
                symbol_allocs=OrderedDict([('AXP', 0.0), ('HPQ', 0.0), ('IBM', 0.0), ('HNZ', 1.0)]),
                start_val=1000000),
            outputs=dict(
                cum_ret=0.198105963655,
                avg_daily_ret=0.000763106152672,
                sharpe_ratio=1.30798398744),
            description="Wiki example 2"
        ),
        PortfolioTestCase(
            inputs=dict(
                start_date='2010-06-01',
                end_date='2010-12-31',
                symbol_allocs=OrderedDict([('GOOG', 0.2), ('AAPL', 0.3), ('GLD', 0.4), ('XOM', 0.1)]),
                start_val=1000000),
            outputs=dict(
                cum_ret=0.205113938792,
                avg_daily_ret=0.00129586924366,
                sharpe_ratio=2.21259766672),
            description="Wiki example 3: Six month range"
        ),
        PortfolioTestCase(
            inputs=dict(
                start_date='2010-01-01',
                end_date='2013-05-31',
                symbol_allocs=OrderedDict([('AXP', 0.3), ('HPQ', 0.5), ('IBM', 0.1), ('GOOG', 0.1)]),
                start_val=1000000),
            outputs=dict(
                cum_ret=-0.110888530433,
                avg_daily_ret=-6.50814806831e-05,
                sharpe_ratio=-0.0704694718385),
            description="Normalization check"
        ),
        PortfolioTestCase(
            inputs=dict(
                start_date='2010-01-01',
                end_date='2010-01-31',
                symbol_allocs=OrderedDict([('AXP', 0.9), ('HPQ', 0.0), ('IBM', 0.1), ('GOOG', 0.0)]),
                start_val=1000000),
            outputs=dict(
                cum_ret=-0.0758725033871,
                avg_daily_ret=-0.00411578300489,
                sharpe_ratio=-2.84503813366),
            description="One month range"
        ),
        PortfolioTestCase(
            inputs=dict(
                start_date='2011-01-01',
                end_date='2011-12-31',
                symbol_allocs=OrderedDict([('WFR', 0.25), ('ANR', 0.25), ('MWW', 0.25), ('FSLR', 0.25)]),
                start_val=1000000),
            outputs=dict(
                cum_ret=-0.686004563165,
                avg_daily_ret=-0.00405018240566,
                sharpe_ratio=-1.93664660013),
            description="Low Sharpe ratio"
        ),
        PortfolioTestCase(
            inputs=dict(
                start_date='2010-01-01',
                end_date='2010-12-31',
                symbol_allocs=OrderedDict([('AXP', 0.0), ('HPQ', 1.0), ('IBM', 0.0), ('HNZ', 0.0)]),
                start_val=1000000),
            outputs=dict(
                cum_ret=-0.191620333598,
                avg_daily_ret=-0.000718040989619,
                sharpe_ratio=-0.71237182415),
            description="All your eggs in one basket"
        ),
        PortfolioTestCase(
            inputs=dict(
                start_date='2006-01-03',
                end_date='2008-01-02',
                symbol_allocs=OrderedDict([('MMM', 0.0), ('MO', 0.9), ('MSFT', 0.1), ('INTC', 0.0)]),
                start_val=1000000),
            outputs=dict(
                cum_ret=0.43732715979,
                avg_daily_ret=0.00076948918955,
                sharpe_ratio=1.26449481371),
            description="Two year range"
        ),
        PortfolioTestCase(
            inputs=dict(
                start_date='2006-01-03',
                end_date='2008-01-02',
                symbol_allocs=OrderedDict([('MMM', 0.2), ('MO', 0.1), ('MSFT', 0.1), ('INTC', 0.1),
                                           ('AXP', 0.1), ('HPQ', 0.2), ('IBM', 0.1), ('HNZ', 0.1)]),
                start_val=1000000),
            outputs=dict(
                cum_ret=0.3268116186758909,
                avg_daily_ret=0.00076948918955,
                sharpe_ratio=1.08628110936518),
            description="Two year range"
        ),
        PortfolioTestCase(
            inputs=dict(
                start_date='2006-01-03',
                end_date='2008-01-02',
                symbol_allocs=OrderedDict([('MMM', 0.2), ('MO', 0.1)]),
                start_val=1000000),
            outputs=dict(
                cum_ret=0.21294731291440505,
                avg_daily_ret=0.00076948918955,
                sharpe_ratio=0.7396973179037072),
            description="Two year range"
        )
    ]
    results = np.zeros(shape=(len(portfolio_test_cases),), dtype=np.bool)
    for test in range(len(portfolio_test_cases)):
        inputs = portfolio_test_cases[test][0]
        start_value = inputs["start_val"]
        symbol_allocs = inputs["symbol_allocs"]
        symbols = []
        allocs = []
        for key, val in symbol_allocs.items():
            symbols.append(key)
            allocs.append(val)
        outputs = portfolio_test_cases[test][1]

        dates = pd.date_range(inputs['start_date'], inputs['end_date'])
        prices_all = get_data(symbols, dates)  # automatically adds SPY
        prices = prices_all[symbols]  # only portfolio symbols

        try:
            result_dict = get_portfolio_value(prices_dataframe=prices, allocs=allocs, start_val=start_value)
        except Exception as err:
            print("Exception when attempting to get portfolio values. \n", err)
            break

        my_sharpe_ratio = result_dict["sharpe_ratio"]
        average_daily_return = result_dict["average_daily_returns"]
        cumulative_return = result_dict["cumulative_returns"]

        tests = np.zeros(shape=(3,), dtype=np.bool)
        tests[0] = np.isclose(outputs['sharpe_ratio'], my_sharpe_ratio, atol=0.01)
        tests[1] = np.isclose(outputs['avg_daily_ret'], average_daily_return, atol=0.01)
        tests[2] = np.isclose(outputs['cum_ret'], cumulative_return, atol=0.01)
        results[test] = np.all(tests)

        print("\nTest {}".format(test))
        print("\tSharpe Ratio: {sr}".format(sr="Failed" if not tests[0] else "Pass"))
        if not tests[0]:
            print(f"\t\tTheir Sharpe Ratio: {outputs['sharpe_ratio']}")
            print(f"\t\tMy Sharpe Ratio: {my_sharpe_ratio}")

        print("\tAverage Period Return: {apr}".format(apr="Failed" if not tests[1] else "Pass"))
        if not tests[1]:
            print(f"\t\tTheir Average Daily Return: {outputs['avg_daily_ret']}")
            print(f"\t\tMy Average Daily Return: {average_daily_return}")

        print("\tCumulative Return: {cr}".format(cr="Failed" if not tests[2] else "Pass"))
        if not tests[2]:
            print(f"\t\tTheir Cumulative Return: {outputs['cum_ret']}")
            print(f"\t\tMy Cumulative Return: {cumulative_return}")

    return results


def tests_2():
    from collections import namedtuple
    OptimizationTestCase = namedtuple(
        "OptimizationTestCase", ["inputs", "outputs", "description"]
    )
    optimization_test_cases = [
        OptimizationTestCase(
            inputs=dict(
                start_date=str2dt("2010-01-01"),
                end_date=str2dt("2010-12-31"),
                symbols=["GOOG", "AAPL", "GLD", "XOM"],
            ),
            outputs=dict(allocs=[0.0, 0.4, 0.6, 0.0]),
            description="Wiki example 1",
        ),
        OptimizationTestCase(
            inputs=dict(
                start_date=str2dt("2004-01-01"),
                end_date=str2dt("2006-01-01"),
                symbols=["AXP", "HPQ", "IBM", "HNZ"],
            ),
            outputs=dict(allocs=[0.78, 0.22, 0.0, 0.0]),
            description="Wiki example 2",
        ),
        OptimizationTestCase(
            inputs=dict(
                start_date=str2dt("2004-12-01"),
                end_date=str2dt("2006-05-31"),
                symbols=["YHOO", "XOM", "GLD", "HNZ"],
            ),
            outputs=dict(allocs=[0.0, 0.07, 0.59, 0.34]),
            description="Wiki example 3",
        ),
        OptimizationTestCase(
            inputs=dict(
                start_date=str2dt("2005-12-01"),
                end_date=str2dt("2006-05-31"),
                symbols=["YHOO", "HPQ", "GLD", "HNZ"],
            ),
            outputs=dict(allocs=[0.0, 0.1, 0.25, 0.65]),
            description="Wiki example 4",
        ),
        OptimizationTestCase(
            inputs=dict(
                start_date=str2dt("2005-12-01"),
                end_date=str2dt("2007-05-31"),
                symbols=["MSFT", "HPQ", "GLD", "HNZ"],
            ),
            outputs=dict(allocs=[0.0, 0.27, 0.11, 0.62]),
            description="MSFT vs HPQ",
        ),
        OptimizationTestCase(
            inputs=dict(
                start_date=str2dt("2006-05-31"),
                end_date=str2dt("2007-05-31"),
                symbols=["MSFT", "AAPL", "GLD", "HNZ"],
            ),
            outputs=dict(allocs=[0.42, 0.32, 0.0, 0.26]),
            description="MSFT vs AAPL",
        ),
        OptimizationTestCase(
            inputs=dict(
                start_date=str2dt("2011-01-01"),
                end_date=str2dt("2011-12-31"),
                symbols=["AAPL", "GLD", "GOOG", "XOM"],
            ),
            outputs=dict(allocs=[0.46, 0.37, 0.0, 0.17]),
            description="Wiki example 1 in 2011",
        ),
        OptimizationTestCase(
            inputs=dict(
                start_date=str2dt("2010-01-01"),
                end_date=str2dt("2010-12-31"),
                symbols=["AXP", "HPQ", "IBM", "HNZ"],
            ),
            outputs=dict(allocs=[0.0, 0.0, 0.0, 1.0]),
            description="Year of the HNZ",
        ),
    ]
    results = np.zeros(shape=(len(optimization_test_cases, )), dtype=np.bool)
    for opt_test in range(len(optimization_test_cases)):
        inputs = optimization_test_cases[opt_test][0]
        symbols = inputs["symbols"]
        try:
            my_allocations, _, _, _, _ = optimize_portfolio(sd=inputs['start_date'],
                                                            ed=inputs['end_date'],
                                                            syms=symbols,
                                                            gen_plot=False)
        except Exception as err:
            print("Exception during optimize portfolio. \n", err)
            break

        check = np.allclose(optimization_test_cases[opt_test][1]["allocs"], my_allocations, atol=0.01)
        print("\nTest {}".format(opt_test))
        results[opt_test] = check
        print("\tOptimized Allocations: {sr}".format(sr="Failed" if not check else "Pass"))
        if not check:
            print(f"\t\tTheir Optimized Allocations: {optimization_test_cases[opt_test][1]['allocs']}")
            print(f"\t\tMy Optimized Allocations: {my_allocations}")

    return results


def run_side_tests():
    print("Beginning Tests for \n\t\tCumulative Return"
          "\n\t\tSharpe Ratio"
          "\n\t\tAverage Daily Value\n")

    test_one_results = tests_1()
    test_one_passed_array = test_one_results[test_one_results]
    print("Results: \n\tPassed: {}\n\tFailed: {}\n".format(test_one_passed_array.size,
                                                           (test_one_results.size - test_one_passed_array.size)))

    print("Beginning Tests for \n\t Optimized Allocations")
    test_two_results = tests_2()
    test_two_passed_array = test_two_results[test_two_results]
    print("Results: \n\tPassed: {}\n\tFailed: {}\n".format(test_two_passed_array.size,
                                                           (test_two_results.size - test_two_passed_array.size)))

    return


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    run_side_tests()
    # print("Finished")
