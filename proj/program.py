import time as t
import datetime as dt
import utils
#from course1_math_and_python.root import main as m, title
#from  course2_supervised_learning.root import main as m, title
#from  course3_structure_in_data.root import main as m, title
from  course4_data_conclusions.root import main as m, title

def main(args=None):

  utils.PRINT.HEADER(title())
  print('STARTED ', dt.datetime.now())
  start = t.time()
  m()
  end = t.time()
  utils.PRINT.HEADER('DONE in {}s ({}m)'.format(round(end-start, 2), round((end-start)/60, 2)))

  return


if __name__=='__main__':
  main()