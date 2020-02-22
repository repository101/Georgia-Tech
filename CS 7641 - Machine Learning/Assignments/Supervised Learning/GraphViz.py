import os
import sys


def Graph(dotData, filename, datasetName='MNIST'):
    try:
        if dotData is not None and filename is not None:
            import graphviz
        else:
            raise Exception
        second_graph = graphviz.Source(dotData)
        second_graph.render("{}_{}".format(datasetName, filename),
                            directory="DecisionTree_Results/", format="pdf")
    
    except Exception as GraphVizException:
        print("Exception occurred while executing 'Graph' within GraphViz.py. \n", GraphVizException)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
